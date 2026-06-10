#pragma once

#include "floorplan/floorplan_state.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace floorplan {

// ─── Bookshelf format parser ─────────────────────────────────────────────
//
// Bookshelf is the standard EDA benchmark format (MCNC / GSRC).
//   .aux   : top-level file (lists .nodes, .nets, .pl, .wts, .scl)
//   .nodes : module dimensions  (format: "o<name>\t<width>\t<height>")
//   .nets  : netlist connections (format: "NetDegree : <n>" then "o<name>" per line)
//   .pl    : initial placement (optional, we ignore for floorplanning)
//
// This is a simplified parser for Phase 1. It assumes:
//   - Module names are "o1", "o2", ... (non-terminal modules)
//   - No terminals / pads (purely block packing)
//   - No .scl (no rows, this is floorplanning not placement)
//
inline FloorplanState
FloorplanState::from_bookshelf(const std::string& aux_path) {
    std::ifstream aux(aux_path);
    if (!aux) throw std::runtime_error("Cannot open .aux: " + aux_path);

    std::string base_dir;
    auto last_slash = aux_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        base_dir = aux_path.substr(0, last_slash + 1);
    }

    std::string nodes_file, nets_file;

    std::string line;
    while (std::getline(aux, line)) {
        // Skip comments and empty lines
        auto first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos) continue;
        if (line[first_non_space] == '#') continue;

        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) tokens.push_back(token);
        if (tokens.empty()) continue;

        // .aux format: "BenchMark : <name>" or "<nodes_file> <nets_file> ..."
        if (tokens[0] == "BenchMark" || tokens[0] == "Benchmark") continue;

        // Collect file paths by extension
        for (const auto& t : tokens) {
            if (t.size() >= 6 && t.substr(t.size() - 6) == ".nodes") {
                nodes_file = base_dir + t;
            } else if (t.size() >= 5 && t.substr(t.size() - 5) == ".nets") {
                nets_file = base_dir + t;
            }
        }
    }

    if (nodes_file.empty() || nets_file.empty()) {
        throw std::runtime_error("Bookshelf .aux missing .nodes or .nets reference: " + aux_path);
    }

    FloorplanState s;

    // ── Parse .nodes ──
    {
        std::ifstream nodes(nodes_file);
        if (!nodes) throw std::runtime_error("Cannot open .nodes: " + nodes_file);

        std::string keyword, name;
        double w, h;
        while (nodes >> keyword) {
            if (keyword == "UCLA" || keyword == "#") {
                std::getline(nodes, line);  // skip header / comment
                continue;
            }
            if (keyword == "NumNodes" || keyword == "NumTerminals") {
                std::getline(nodes, line);  // skip count lines
                continue;
            }
            // keyword is the module name
            name = keyword;
            if (!(nodes >> w >> h)) break;

            std::size_t id = s.mutable_modules().size();
            s.mutable_modules().push_back({id, w, h, w * h, {}});
        }
        s.mutable_is_placed().assign(s.num_modules(), false);
    }

    // Build name -> module_id map
    std::unordered_map<std::string, std::size_t> name_to_id;
    for (std::size_t i = 0; i < s.num_modules(); ++i) {
        // Module names in .nodes are typically "o1", "o2", ...
        name_to_id["o" + std::to_string(i + 1)] = i;
    }

    // ── Parse .nets ──
    {
        std::ifstream nets(nets_file);
        if (!nets) throw std::runtime_error("Cannot open .nets: " + nets_file);

        std::string keyword;
        while (nets >> keyword) {
            if (keyword == "UCLA" || keyword == "#") {
                std::getline(nets, line);
                continue;
            }
            if (keyword != "NetDegree") {
                std::getline(nets, line);
                continue;
            }

            std::string colon;
            std::size_t degree = 0;
            nets >> colon >> degree;
            if (colon != ":") continue;

            std::vector<std::size_t> net;
            net.reserve(degree);
            for (std::size_t i = 0; i < degree; ++i) {
                std::string term_name;
                if (!(nets >> term_name)) break;

                // Strip optional pin offset (e.g., "o1 : N" or just "o1")
                auto colon_pos = term_name.find(':');
                if (colon_pos != std::string::npos) {
                    term_name = term_name.substr(0, colon_pos);
                }
                // Trim trailing whitespace
                while (!term_name.empty() && std::isspace(term_name.back())) {
                    term_name.pop_back();
                }

                auto it = name_to_id.find(term_name);
                if (it != name_to_id.end()) {
                    std::size_t mod_id = it->second;
                    // Avoid duplicate module IDs in the same net
                    bool already_in = false;
                    for (std::size_t mid : net) {
                        if (mid == mod_id) { already_in = true; break; }
                    }
                    if (!already_in) net.push_back(mod_id);
                }
            }
            if (net.size() >= 2) {
                s.mutable_nets().push_back(std::move(net));
            }
        }
    }

    return s;
}

} // namespace floorplan
