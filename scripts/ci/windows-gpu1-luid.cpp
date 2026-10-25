// Host metadata only. SDK declarations, not handwritten ABI layouts or enum values.
// https://learn.microsoft.com/windows-hardware/drivers/ddi/d3dkmthk/nf-d3dkmthk-d3dkmtenumadapters2
#include <windows.h>
#include <d3dkmthk.h>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Adapters {
    std::vector<D3DKMT_ADAPTERINFO> rows;
    bool close() noexcept {
        bool ok = true;
        for (auto &row : rows) {
            if (row.hAdapter) {
                D3DKMT_CLOSEADAPTER arg{};
                arg.hAdapter = row.hAdapter;
                if (D3DKMTCloseAdapter(&arg) != 0) ok = false;
                row.hAdapter = 0;
            }
        }
        return ok;
    }
    ~Adapters() { close(); }
};

unsigned hex_arg(const char *input, unsigned maximum) {
    const std::string value(input);
    unsigned result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result > maximum)
        throw std::runtime_error("invalid PCI address argument");
    return result;
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("expected PCI bus device function");
        const auto bus = hex_arg(argv[1], 255), device = hex_arg(argv[2], 31), function = hex_arg(argv[3], 7);
        D3DKMT_ENUMADAPTERS2 enumeration{};
        if (D3DKMTEnumAdapters2(&enumeration) != 0 || enumeration.NumAdapters == 0 || enumeration.NumAdapters > 64)
            throw std::runtime_error("adapter enumeration size unavailable");
        Adapters adapters;
        adapters.rows.resize(enumeration.NumAdapters);
        enumeration.pAdapters = adapters.rows.data();
        if (D3DKMTEnumAdapters2(&enumeration) != 0 || enumeration.NumAdapters > adapters.rows.size())
            throw std::runtime_error("adapter enumeration failed or changed");
        unsigned matches = 0;
        LUID selected{};
        for (ULONG i = 0; i < enumeration.NumAdapters; ++i) {
            D3DKMT_ADAPTERADDRESS address{};
            D3DKMT_QUERYADAPTERINFO query{};
            query.hAdapter = adapters.rows[i].hAdapter;
            query.Type = KMTQAITYPE_ADAPTERADDRESS;
            query.pPrivateDriverData = &address;
            query.PrivateDriverDataSize = sizeof(address);
            if (D3DKMTQueryAdapterInfo(&query) != 0)
                throw std::runtime_error("adapter PCI metadata unavailable");
            if (address.BusNumber != bus || address.DeviceNumber != device || address.FunctionNumber != function) continue;
            D3DKMT_PHYSICAL_ADAPTER_COUNT physical{};
            query.Type = KMTQAITYPE_PHYSICALADAPTERCOUNT;
            query.pPrivateDriverData = &physical;
            query.PrivateDriverDataSize = sizeof(physical);
            if (D3DKMTQueryAdapterInfo(&query) != 0 || physical.Count != 1)
                throw std::runtime_error("selected adapter physical mapping unavailable or linked");
            selected = adapters.rows[i].AdapterLuid;
            ++matches;
        }
        if (!adapters.close()) throw std::runtime_error("adapter metadata handle close failed");
        if (matches != 1) throw std::runtime_error("PCI to LUID match missing or ambiguous");
        std::cout << "{\"status\":\"mapped\",\"bus\":" << bus << ",\"device\":" << device
                  << ",\"function\":" << function << ",\"luidHigh\":" << static_cast<unsigned long>(selected.HighPart)
                  << ",\"luidLow\":" << selected.LowPart << ",\"counterLuid\":\"luid_0x"
                  << std::hex << std::setfill('0') << std::setw(8) << static_cast<unsigned long>(selected.HighPart)
                  << "_0x" << std::setw(8) << selected.LowPart << "\"}\n";
        return 0;
    } catch (const std::exception &error) {
        // All exception strings above are fixed diagnostics, never process metadata or secrets.
        std::cout << "{\"status\":\"unknown\",\"error\":\"" << error.what() << "\"}\n";
        return 1;
    }
}
