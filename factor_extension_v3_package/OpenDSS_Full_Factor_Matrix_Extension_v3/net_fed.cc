/*
 * net_fed.cc -- production ns-3.35 / HELICS 3.6.1 telemetry network federate.
 *
 * Each telemetry channel owns a real ns-3 point-to-point link containing two
 * NetDevices, a DropTailQueue, a DataRate serializer, and a propagation-delay
 * channel.  The impairment table changes link state by event.  Bernoulli packet
 * loss is realized independently at ingress with ns-3 RNG streams so a packet's
 * event-specific loss probability cannot change while it is in flight.
 *
 * Input wire format:
 *   <channel_id><TAB>{"schema":"twin.telemetry.v1",...,"event_id":N}
 *
 * Output wire format:
 *   the unchanged JSON payload, delivered to twin_fed/in at ns-3 delivery time.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/helics-id-tag.h"  // Waf dependency marker for contrib/helics

#include <helics/application_api/MessageFederate.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Impairment
{
    double dropoutProbability{0.0};
    double latencySeconds{0.0};
    double bandwidthBps{1.0e12};
};

using Key = std::pair<std::uint64_t, std::uint32_t>;

struct PatternTable
{
    std::map<Key, Impairment> values;
    std::uint64_t events{0};
    std::uint32_t channels{0};
};

struct Counters
{
    std::uint64_t received{0};
    std::uint64_t delivered{0};
    std::uint64_t droppedRandom{0};
    std::uint64_t droppedStarved{0};
    std::uint64_t droppedQueue{0};
    std::uint64_t malformed{0};
    std::uint64_t bytesReceived{0};
    std::uint64_t bytesDelivered{0};
};

struct DeliveryContext
{
    helics::Endpoint* endpoint{nullptr};
    std::string destination;
    Counters* counters{nullptr};
    double zeroLatencyTolerance{1.0e-6};
    std::map<std::uint64_t, double> ingressTime;
};

struct Link
{
    ns3::Ptr<ns3::PointToPointNetDevice> sender;
    ns3::Ptr<ns3::PointToPointNetDevice> receiver;
    ns3::Ptr<ns3::PointToPointChannel> channel;
    ns3::Ptr<ns3::UniformRandomVariable> lossRandom;
};

std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ','))
    {
        fields.push_back(field);
    }
    return fields;
}

PatternTable LoadPatterns(const std::string& path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("cannot open pattern CSV: " + path);
    }
    std::string line;
    if (!std::getline(input, line))
    {
        throw std::runtime_error("pattern CSV is empty: " + path);
    }
    const std::vector<std::string> expected{
        "event_id", "channel_id", "pi", "delta_s", "bandwidth_bps"};
    if (SplitCsvLine(line) != expected)
    {
        throw std::runtime_error(
            "pattern CSV header must be event_id,channel_id,pi,delta_s,bandwidth_bps");
    }

    PatternTable table;
    std::size_t lineNumber = 1;
    while (std::getline(input, line))
    {
        ++lineNumber;
        if (line.empty())
        {
            continue;
        }
        const auto fields = SplitCsvLine(line);
        if (fields.size() != 5)
        {
            throw std::runtime_error("malformed pattern row at line " +
                                     std::to_string(lineNumber));
        }
        const auto eventId = static_cast<std::uint64_t>(std::stoull(fields[0]));
        const auto channelId = static_cast<std::uint32_t>(std::stoul(fields[1]));
        Impairment item;
        item.dropoutProbability = std::stod(fields[2]);
        item.latencySeconds = std::stod(fields[3]);
        item.bandwidthBps = std::stod(fields[4]);
        if (!std::isfinite(item.dropoutProbability) ||
            !std::isfinite(item.latencySeconds) ||
            !std::isfinite(item.bandwidthBps) ||
            item.dropoutProbability < 0.0 || item.dropoutProbability > 1.0 ||
            item.latencySeconds < 0.0 || item.bandwidthBps < 0.0)
        {
            throw std::runtime_error("out-of-range pattern value at line " +
                                     std::to_string(lineNumber));
        }
        if (!table.values.emplace(Key{eventId, channelId}, item).second)
        {
            throw std::runtime_error("duplicate event/channel pattern at line " +
                                     std::to_string(lineNumber));
        }
        table.events = std::max(table.events, eventId + 1);
        table.channels = std::max(table.channels, channelId + 1);
    }
    if (table.values.empty())
    {
        throw std::runtime_error("pattern CSV contains no data rows");
    }
    return table;
}

std::pair<std::uint32_t, std::string> DecodeEnvelope(const std::string& message)
{
    const auto separator = message.find('\t');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= message.size())
    {
        throw std::runtime_error("incoming message lacks <channel_id><TAB><payload>");
    }
    std::size_t consumed = 0;
    const unsigned long parsed = std::stoul(message.substr(0, separator), &consumed);
    if (consumed != separator || parsed > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::runtime_error("incoming channel_id is invalid");
    }
    return {static_cast<std::uint32_t>(parsed), message.substr(separator + 1)};
}

std::uint64_t ExtractUnsignedJson(const std::string& payload, const std::string& key)
{
    const std::string token = "\"" + key + "\"";
    const auto keyPosition = payload.find(token);
    if (keyPosition == std::string::npos)
    {
        throw std::runtime_error("telemetry payload is missing JSON field " + key);
    }
    const auto colon = payload.find(':', keyPosition + token.size());
    if (colon == std::string::npos)
    {
        throw std::runtime_error("telemetry payload has malformed JSON field " + key);
    }
    const auto first = payload.find_first_not_of(" \t\r\n", colon + 1);
    if (first == std::string::npos || payload[first] == '-')
    {
        throw std::runtime_error("telemetry JSON field " + key + " must be unsigned");
    }
    std::size_t consumed = 0;
    const auto value = std::stoull(payload.substr(first), &consumed);
    if (consumed == 0)
    {
        throw std::runtime_error("telemetry JSON field " + key + " is invalid");
    }
    return value;
}

bool ReceivePacket(std::uint32_t channelId,
                   DeliveryContext* context,
                   ns3::Ptr<ns3::NetDevice>,
                   ns3::Ptr<const ns3::Packet> packet,
                   std::uint16_t,
                   const ns3::Address&)
{
    std::vector<std::uint8_t> bytes(packet->GetSize());
    if (!bytes.empty())
    {
        packet->CopyData(bytes.data(), static_cast<std::uint32_t>(bytes.size()));
    }
    double deliveryTime = ns3::Simulator::Now().GetSeconds();
    const auto ingress = context->ingressTime.find(packet->GetUid());
    if (ingress != context->ingressTime.end())
    {
        if (deliveryTime - ingress->second <= context->zeroLatencyTolerance)
        {
            deliveryTime = ingress->second;
        }
        context->ingressTime.erase(ingress);
    }

    const char* data = bytes.empty()
        ? ""
        : reinterpret_cast<const char*>(bytes.data());
    context->endpoint->sendToAt(
        data, bytes.size(), context->destination, helics::Time(deliveryTime));
    ++context->counters->delivered;
    context->counters->bytesDelivered += bytes.size();
    return true;
}

void AdvanceNs3To(double target)
{
    const double now = ns3::Simulator::Now().GetSeconds();
    if (target <= now + 1.0e-12)
    {
        return;
    }
    ns3::Simulator::Stop(ns3::Seconds(target - now));
    ns3::Simulator::Run();
}

std::vector<Link> BuildLinks(std::uint32_t count,
                             DeliveryContext* delivery,
                             std::uint64_t streamBase)
{
    ns3::NodeContainer nodes;
    nodes.Create(2 * count);
    std::vector<Link> links;
    links.reserve(count);
    for (std::uint32_t channelId = 0; channelId < count; ++channelId)
    {
        ns3::PointToPointHelper helper;
        helper.SetDeviceAttribute(
            "DataRate", ns3::DataRateValue(ns3::DataRate(UINT64_C(1000000000000))));
        helper.SetChannelAttribute("Delay", ns3::TimeValue(ns3::Seconds(0.0)));
        helper.SetQueue("ns3::DropTailQueue", "MaxSize", ns3::StringValue("1000p"));
        const auto devices = helper.Install(nodes.Get(2 * channelId), nodes.Get(2 * channelId + 1));
        Link link;
        link.sender = ns3::DynamicCast<ns3::PointToPointNetDevice>(devices.Get(0));
        link.receiver = ns3::DynamicCast<ns3::PointToPointNetDevice>(devices.Get(1));
        link.channel = ns3::DynamicCast<ns3::PointToPointChannel>(link.sender->GetChannel());
        if (link.sender == nullptr || link.receiver == nullptr || link.channel == nullptr)
        {
            throw std::runtime_error("failed to create point-to-point link");
        }
        link.receiver->SetReceiveCallback(
            ns3::MakeBoundCallback(&ReceivePacket, channelId, delivery));
        link.lossRandom = ns3::CreateObject<ns3::UniformRandomVariable>();
        link.lossRandom->SetStream(static_cast<std::int64_t>(streamBase + channelId));
        links.push_back(link);
    }
    return links;
}

void ValidateCoverage(const PatternTable& patterns, std::uint32_t nTelemetry)
{
    if (nTelemetry == 0 || nTelemetry > patterns.channels)
    {
        throw std::runtime_error("nTelemetry is outside the pattern-table channel range");
    }
    for (std::uint64_t event = 0; event < patterns.events; ++event)
    {
        for (std::uint32_t channel = 0; channel < nTelemetry; ++channel)
        {
            if (patterns.values.find(Key{event, channel}) == patterns.values.end())
            {
                throw std::runtime_error(
                    "pattern table is missing event=" + std::to_string(event) +
                    " channel=" + std::to_string(channel));
            }
        }
    }
}

void WriteMeta(const std::string& path,
               const std::string& status,
               const PatternTable& patterns,
               std::uint32_t nTelemetry,
               double timeStep,
               double helicsTimeDelta,
               double stopTime,
               double tailTime,
               double bMin,
               double bandwidthCapBps,
               const std::string& bandwidthLevel,
               std::uint64_t seed,
               std::uint64_t run,
               const Counters& counters)
{
    const std::string temporary = path + ".tmp";
    std::ofstream output(temporary);
    if (!output)
    {
        throw std::runtime_error("cannot write network meta: " + path);
    }
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"net.run.meta.v1\",\n"
           << "  \"status\": \"" << status << "\",\n"
           << "  \"events\": " << patterns.events << ",\n"
           << "  \"pattern_channels\": " << patterns.channels << ",\n"
           << "  \"telemetry_channels\": " << nTelemetry << ",\n"
           << "  \"time_step\": " << timeStep << ",\n"
           << "  \"helics_time_delta\": " << helicsTimeDelta << ",\n"
           << "  \"stage_offset\": " << helicsTimeDelta << ",\n"
           << "  \"stop_time\": " << stopTime << ",\n"
           << "  \"tail_time\": " << tailTime << ",\n"
           << "  \"b_min\": " << bMin << ",\n"
           << "  \"bandwidth_cap_bps\": " << bandwidthCapBps << ",\n"
           << "  \"bandwidth_level\": \"" << bandwidthLevel << "\",\n"
           << "  \"seed\": " << seed << ",\n"
           << "  \"run\": " << run << ",\n"
           << "  \"counts\": {\n"
           << "    \"received\": " << counters.received << ",\n"
           << "    \"delivered\": " << counters.delivered << ",\n"
           << "    \"dropped_random\": " << counters.droppedRandom << ",\n"
           << "    \"dropped_starved\": " << counters.droppedStarved << ",\n"
           << "    \"dropped_queue\": " << counters.droppedQueue << ",\n"
           << "    \"malformed\": " << counters.malformed << ",\n"
           << "    \"bytes_received\": " << counters.bytesReceived << ",\n"
           << "    \"bytes_delivered\": " << counters.bytesDelivered << "\n"
           << "  }\n"
           << "}\n";
    output.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
    {
        throw std::runtime_error("cannot atomically replace network meta: " + path);
    }
}

} // namespace

int main(int argc, char* argv[])
{
    std::string patternsPath{"patterns.csv"};
    std::string metaPath{"runs/net/meta.json"};
    std::string federateName{"net_fed"};
    std::string inputEndpoint{"net_fed/in"};
    std::string outputEndpoint{"net_fed/out"};
    std::string destination{"twin_fed/in"};
    std::string upstreamFederate{"power_fed"};
    std::string coreType{"zmq"};
    std::string coreInit{"--federates=1"};
    std::string bandwidthLevel{"bw04_oracle"};
    double timeStep = 1.0;
    double helicsTimeDelta = 1.0e-6;
    double stopTime = 0.0;
    double tailTime = 30.0;
    double bMin = 1.0;
    double bandwidthCapBps = 1.0e12;
    double zeroLatencyTolerance = 1.0e-6;
    std::uint32_t nTelemetry = 45;
    std::uint32_t stepsPerEvent = 12;
    std::uint64_t seed = 1;
    std::uint64_t run = 1;

    ns3::CommandLine command(__FILE__);
    command.AddValue("patterns", "Long-form impairment CSV", patternsPath);
    command.AddValue("meta", "Network run metadata JSON", metaPath);
    command.AddValue("federateName", "HELICS federate name", federateName);
    command.AddValue("inputEndpoint", "Global HELICS ingress endpoint", inputEndpoint);
    command.AddValue("outputEndpoint", "Global HELICS egress endpoint", outputEndpoint);
    command.AddValue("destination", "HELICS destination endpoint", destination);
    command.AddValue("upstreamFederate", "Explicit HELICS time dependency", upstreamFederate);
    command.AddValue("coreType", "HELICS core type", coreType);
    command.AddValue("coreInit", "HELICS core initialization string", coreInit);
    command.AddValue("timeStep", "Logical update interval", timeStep);
    command.AddValue("helicsTimeDelta", "HELICS timing resolution", helicsTimeDelta);
    command.AddValue("stopTime", "Final logical time; 0 derives from patterns", stopTime);
    command.AddValue("tailTime", "Network flush interval", tailTime);
    command.AddValue("stepsPerEvent", "Logical updates per event", stepsPerEvent);
    command.AddValue("nTelemetry", "Number of physical telemetry channels", nTelemetry);
    command.AddValue("bMin", "Bandwidth at/below which a channel is starved", bMin);
    command.AddValue(
        "bandwidthCapBps",
        "Cross-factor cap applied to every pattern bandwidth",
        bandwidthCapBps);
    command.AddValue(
        "bandwidthLevel",
        "Predeclared bandwidth-level identifier",
        bandwidthLevel);
    command.AddValue("zeroLatencyTolerance", "Round negligible latency to current time", zeroLatencyTolerance);
    command.AddValue("seed", "ns-3 RNG seed", seed);
    command.AddValue("run", "ns-3 RNG run/substream", run);
    command.Parse(argc, argv);

    if (timeStep <= 0.0 || helicsTimeDelta <= 0.0 || tailTime < 0.0 ||
        bMin < 0.0 || !std::isfinite(bandwidthCapBps) || bandwidthCapBps <= 0.0 ||
        bandwidthLevel.empty() ||
        bandwidthLevel.find_first_not_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-") !=
            std::string::npos ||
        zeroLatencyTolerance < 0.0 || stepsPerEvent == 0 ||
        nTelemetry == 0 || upstreamFederate.empty() || seed == 0 || run == 0 ||
        seed > std::numeric_limits<std::uint32_t>::max())
    {
        std::cerr << "invalid timing/channel/RNG argument\n";
        return EXIT_FAILURE;
    }

    std::unique_ptr<helics::MessageFederate> federate;
    Counters counters;
    PatternTable patterns;
    bool completed = false;
    try
    {
        patterns = LoadPatterns(patternsPath);
        ValidateCoverage(patterns, nTelemetry);
        const double derivedStop = patterns.events * stepsPerEvent * timeStep;
        if (stopTime <= 0.0)
        {
            stopTime = derivedStop;
        }
        const double rawSteps = stopTime / timeStep;
        const auto totalSteps = static_cast<std::uint64_t>(std::llround(rawSteps));
        if (totalSteps == 0 ||
            std::abs(rawSteps - static_cast<double>(totalSteps)) > 1.0e-9 ||
            totalSteps > patterns.events * stepsPerEvent)
        {
            throw std::runtime_error("stopTime must select 1..patterns*stepsPerEvent updates");
        }

        ns3::RngSeedManager::SetSeed(static_cast<std::uint32_t>(seed));
        ns3::RngSeedManager::SetRun(run);

        if (coreType != "zmq")
        {
            throw std::runtime_error(
                "this experiment requires HELICS core type zmq; received: " + coreType);
        }
        helics::FederateInfo info(helics::CoreType::ZMQ);
        info.coreInitString = coreInit;
        info.setProperty(HELICS_PROPERTY_TIME_DELTA, helicsTimeDelta);
        info.setFlagOption(HELICS_FLAG_UNINTERRUPTIBLE, true);

        federate = std::make_unique<helics::MessageFederate>(federateName, info);
        federate->addDependency(upstreamFederate);
        federate->registerGlobalEndpoint(inputEndpoint, "json");
        federate->registerGlobalEndpoint(outputEndpoint, "json");
        auto& input = federate->getEndpoint(inputEndpoint);
        auto& output = federate->getEndpoint(outputEndpoint);

        DeliveryContext delivery;
        delivery.endpoint = &output;
        delivery.destination = destination;
        delivery.counters = &counters;
        delivery.zeroLatencyTolerance = zeroLatencyTolerance;
        auto links = BuildLinks(nTelemetry, &delivery, 1000);
        WriteMeta(metaPath, "connecting", patterns, nTelemetry, timeStep,
                  helicsTimeDelta, stopTime,
                  tailTime, bMin, bandwidthCapBps, bandwidthLevel,
                  seed, run, counters);

        federate->enterExecutingMode();
        WriteMeta(metaPath, "running", patterns, nTelemetry, timeStep,
                  helicsTimeDelta, stopTime,
                  tailTime, bMin, bandwidthCapBps, bandwidthLevel,
                  seed, run, counters);

        for (std::uint64_t logicalStep = 0; logicalStep < totalSteps; ++logicalStep)
        {
            const double requested =
                (logicalStep + 1) * timeStep + helicsTimeDelta;
            const double granted = static_cast<double>(
                federate->requestTime(helics::Time(requested)));
            if (std::abs(granted - requested) > 1.0e-9)
            {
                throw std::runtime_error("net_fed received an unexpected nonlogical grant");
            }
            AdvanceNs3To(granted);

            while (input.hasMessage())
            {
                auto message = input.getMessage();
                if (!message)
                {
                    break;
                }
                try
                {
                    const auto messageView = message->to_string();
                    const std::string incoming(messageView.data(), messageView.size());
                    const auto envelope = DecodeEnvelope(incoming);
                    const std::uint32_t channelId = envelope.first;
                    const std::string& payload = envelope.second;
                    const auto payloadChannel = ExtractUnsignedJson(payload, "channel_id");
                    const auto eventId = ExtractUnsignedJson(payload, "event_id");
                    if (channelId >= nTelemetry || payloadChannel != channelId ||
                        eventId >= patterns.events)
                    {
                        throw std::runtime_error("telemetry channel/event is outside configured range");
                    }
                    const auto pattern = patterns.values.find(Key{eventId, channelId});
                    if (pattern == patterns.values.end())
                    {
                        throw std::runtime_error("missing event/channel impairment");
                    }
                    const Impairment& impairment = pattern->second;
                    const double effectiveBandwidth =
                        std::min(impairment.bandwidthBps, bandwidthCapBps);
                    ++counters.received;
                    counters.bytesReceived += payload.size();

                    if (effectiveBandwidth <= bMin)
                    {
                        ++counters.droppedStarved;
                    }
                    else if (links[channelId].lossRandom->GetValue(0.0, 1.0) <
                             impairment.dropoutProbability)
                    {
                        ++counters.droppedRandom;
                    }
                    else
                    {
                        const auto rate = static_cast<std::uint64_t>(
                            std::max(1.0, std::min(
                                effectiveBandwidth,
                                static_cast<double>(std::numeric_limits<std::uint64_t>::max()))));
                        links[channelId].sender->SetDataRate(ns3::DataRate(rate));
                        links[channelId].channel->SetAttribute(
                            "Delay", ns3::TimeValue(ns3::Seconds(impairment.latencySeconds)));
                        ns3::Ptr<ns3::Packet> packet = ns3::Create<ns3::Packet>(
                            reinterpret_cast<const std::uint8_t*>(payload.data()),
                            static_cast<std::uint32_t>(payload.size()));
                        delivery.ingressTime[packet->GetUid()] = granted;
                        if (!links[channelId].sender->Send(
                                packet, links[channelId].receiver->GetAddress(), 0x0800))
                        {
                            delivery.ingressTime.erase(packet->GetUid());
                            ++counters.droppedQueue;
                        }
                    }
                }
                catch (const std::exception& exception)
                {
                    ++counters.malformed;
                    throw std::runtime_error(
                        std::string("invalid telemetry packet: ") + exception.what());
                }
            }

            const double ns3Target = logicalStep + 1 == totalSteps
                ? stopTime + tailTime + helicsTimeDelta
                : requested + timeStep;
            AdvanceNs3To(ns3Target);
        }

        completed = true;
        WriteMeta(metaPath, "complete", patterns, nTelemetry, timeStep,
                  helicsTimeDelta, stopTime,
                  tailTime, bMin, bandwidthCapBps, bandwidthLevel,
                  seed, run, counters);
        federate->finalize();
        federate.reset();
        ns3::Simulator::Destroy();

        std::cout << "net_fed complete: received=" << counters.received
                  << " delivered=" << counters.delivered
                  << " dropped_random=" << counters.droppedRandom
                  << " dropped_starved=" << counters.droppedStarved
                  << " dropped_queue=" << counters.droppedQueue << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "net_fed fatal: " << exception.what() << '\n';
        try
        {
            if (!patterns.values.empty())
            {
                WriteMeta(metaPath, completed ? "complete" : "failed", patterns,
                          nTelemetry, timeStep, helicsTimeDelta, stopTime, tailTime,
                          bMin, bandwidthCapBps, bandwidthLevel,
                          seed, run, counters);
            }
        }
        catch (...)
        {
        }
        if (federate)
        {
            try
            {
                federate->disconnect();
            }
            catch (...)
            {
            }
            federate.reset();
        }
        ns3::Simulator::Destroy();
        return EXIT_FAILURE;
    }
}
