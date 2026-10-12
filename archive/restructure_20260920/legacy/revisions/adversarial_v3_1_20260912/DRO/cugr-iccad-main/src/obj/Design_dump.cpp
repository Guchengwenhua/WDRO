#include "Design.h"
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <regex>

static std::pair<double,double> center_from_boxonlayer(const BoxOnLayer &b) {
    std::ostringstream oss;
    oss << b;
    std::string s = oss.str();
    std::regex re(R"(([-+]?\d+\.?\d*))");
    std::sregex_iterator it(s.begin(), s.end(), re), end;
    std::vector<double> nums;
    for (; it != end; ++it) {
        try { nums.push_back(std::stod((*it)[1].str())); } catch(...) { }
    }
    if (nums.size() >= 4) {
        double minx = nums[0], miny = nums[1], maxx = nums[2], maxy = nums[3];
        return {0.5*(minx+maxx), 0.5*(miny+maxy)};
    }
    // fallback: if only one pair present, use it
    if (nums.size() >= 2) return {nums[0], nums[1]};
    return {0.0, 0.0};
}

void Design::dump_net_terminals_by_degree(const std::string& out_dir, int max_degree, bool normalize) {
    namespace fs = boost::filesystem;
    fs::create_directories(out_dir);

    struct NetRecord { std::string name; int id; std::vector<std::pair<double,double>> pts; };
    std::unordered_map<int, std::vector<NetRecord>> byDegree;

    double global_minx = DBL_MAX, global_miny = DBL_MAX, global_maxx = -DBL_MAX, global_maxy = -DBL_MAX;

    const auto& allNets = this->getAllNets();
    for (const auto &net : allNets) {
        std::string netName = net.getName();
        std::string lname = netName;
        std::transform(lname.begin(), lname.end(), lname.begin(), ::tolower);
        if (lname.find("vdd") != std::string::npos || lname.find("vss") != std::string::npos || lname.find("gnd") != std::string::npos)
            continue;

        const auto& pinRefs = net.getAllPinRefs();
        int degree = (int)pinRefs.size();
        if (degree < 2 || degree > max_degree) continue;

        NetRecord rec;
        rec.name = netName;
        rec.id = net.getIndex();

        for (const auto &pr : pinRefs) {
            double x = 0.0, y = 0.0;
            bool got = false;

            // 1) try Design-level pinShapes
            std::vector<BoxOnLayer> shapes;
            this->getPinShapes(pr, shapes);
            if (!shapes.empty()) {
                auto c = center_from_boxonlayer(shapes[0]);
                x = c.first; y = c.second; got = true;
            }

            // 2) try macro pin shapes
            if (!got) {
                int instIdx = pr.instanceIndex;
                if (instIdx >= 0 && instIdx < (int)instances.size()) {
                    const Instance &inst = instances[instIdx];
                    int midx = inst.getMacroIndex();
                    if (midx >= 0 && midx < (int)macros.size()) {
                        const Macro &mac = macros[midx];
                        const auto &mpins = mac.getPinShapes(pr.pinIndex);
                        if (!mpins.empty()) {
                            auto c = center_from_boxonlayer(mpins[0]);
                            // if macro pin shapes are in macro-local coords, add instance position
                            auto pos = inst.getPosition();
                            x = c.first + (double)pos.x;
                            y = c.second + (double)pos.y;
                            got = true;
                        }
                    }
                }
            }

            // 3) port bbox center
            if (!got) {
                int instIdx = pr.instanceIndex;
                if (instIdx >= 0 && instIdx < (int)instances.size()) {
                    const Instance &inst = instances[instIdx];
                    if (!inst.isCell()) {
                        auto c = center_from_boxonlayer(inst.getPort());
                        x = c.first; y = c.second; got = true;
                    } else {
                        auto pos = inst.getPosition();
                        x = (double)pos.x; y = (double)pos.y; got = true;
                    }
                }
            }

            if (!got) {
                // leave as (0,0)
            } else {
                global_minx = std::min(global_minx, x);
                global_miny = std::min(global_miny, y);
                global_maxx = std::max(global_maxx, x);
                global_maxy = std::max(global_maxy, y);
            }
            rec.pts.emplace_back(x, y);
        }

        if ((int)rec.pts.size() != degree) continue;
        byDegree[degree].push_back(std::move(rec));
    }

    bool can_normalize = normalize && (global_maxx > global_minx) && (global_maxy > global_miny);
    for (auto &kv : byDegree) {
        int degree = kv.first;
        std::string fname = out_dir + "/degree_" + std::to_string(degree) + ".txt";
        std::ofstream ofs(fname, std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) { std::cerr << "Failed to open " << fname << std::endl; continue; }
        ofs << std::fixed << std::setprecision(6);
        for (const auto &rec : kv.second) {
            ofs << "net " << rec.name << " " << rec.id << " " << (int)rec.pts.size() << "\n";
            for (const auto &p : rec.pts) {
                double xx = p.first, yy = p.second;
                if (can_normalize) {
                    double nx = (xx - global_minx) / (global_maxx - global_minx);
                    double ny = (yy - global_miny) / (global_maxy - global_miny);
                    ofs << nx << " " << ny << "\n";
                } else {
                    long long xi = std::llround(xx), yi = std::llround(yy);
                    ofs << xi << " " << yi << "\n";
                }
            }
            ofs << "END\n";
        }
        ofs.close();
    }
}