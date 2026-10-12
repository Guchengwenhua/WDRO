#include "global.h"
#include "obj/Design.h"
#include "gr/GlobalRouter.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char* argv[]) {
    logeol(2);
    log() << "NEW GLOBAL ROUTER CUGR (dump-only mode supported)" << std::endl;
    logeol(2);

    // parse minimal CLI manually
    std::string lef_path, def_path;
    // 输出仍使用 DRO 根目录的 benchmark
    std::string out_root = "/remote-home/Chenlin/OPT/DRO/benchmark";
    int max_degree = 50;
    bool normalize = true;
    bool dump_only = true;
    std::string dataset;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--lef" && i + 1) { lef_path = argv[++i]; continue; }
        if (a == "--def" && i + 1) { def_path = argv[++i]; continue; }
        if (a == "--out_root" && i + 1) { out_root = argv[++i]; continue; }
        if (a == "--max_degree" && i + 1) { max_degree = std::stoi(argv[++i]); continue; }
        if (a == "--normalize" && i + 1) { normalize = (std::stoi(argv[++i]) != 0); continue; }
        if (a == "--dump_only" && i + 1) { dump_only = (std::stoi(argv[++i]) != 0); continue; }
        if (a == "--dataset" && i + 1) { dataset = argv[++i]; continue; }
    }

    if (!dataset.empty()) {
        namespace fs = boost::filesystem;
        fs::path dsdir;
        // 优先查找项目内的 benchmark（cugr-iccad-main/benchmark）
        fs::path proj_bench = fs::path("/remote-home/Chenlin/OPT/DRO/cugr-iccad-main/benchmark") / dataset;
        if (fs::exists(proj_bench) && fs::is_directory(proj_bench)) {
            dsdir = proj_bench;
        } else {
            // 其次查找 out_root 下的 dataset
            fs::path outcand = fs::path(out_root) / dataset;
            if (fs::exists(outcand) && fs::is_directory(outcand)) {
                dsdir = outcand;
            } else {
                // 再尝试当前路径下的 ./benchmark/<dataset>（向上查找）
                fs::path p = fs::current_path();
                bool found = false;
                while (true) {
                    fs::path candp = p / "benchmark" / dataset;
                    if (fs::exists(candp) && fs::is_directory(candp)) {
                        dsdir = candp;
                        found = true;
                        break;
                    }
                    if (p == p.parent_path()) break;
                    p = p.parent_path();
                }
                if (!found) {
                    // 若都没找到，仍在 out_root 下创建目录用于输出
                    if (!fs::exists(outcand)) fs::create_directories(outcand);
                    dsdir = outcand;
                }
            }
        }

        std::cout << "Using dataset directory: " << dsdir.string() << std::endl;
        // 递归查找第一个 lef/def 文件，优先匹配
        for (fs::recursive_directory_iterator it(dsdir); it != fs::recursive_directory_iterator(); ++it) {
            if (!fs::is_regular_file(*it)) continue;
            std::string ext = it->path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".def" && def_path.empty()) def_path = it->path().string();
            if (ext == ".lef" && lef_path.empty()) lef_path = it->path().string();
            if (!lef_path.empty() && !def_path.empty()) break;
        }
        if (lef_path.empty() || def_path.empty()) {
            std::cerr << "Did not find LEF/DEF under " << dsdir.string() << ". Listing files:\n";
            for (fs::directory_iterator it(dsdir); it != fs::directory_iterator(); ++it) {
                std::cerr << "  " << it->path().string() << "\n";
            }
        }
    }

    if (lef_path.empty() || def_path.empty()) {
        std::cerr << "Missing LEF/DEF. Provide --dataset <name> or --lef/--def." << std::endl;
        return 1;
    }

    // Build Parameters from original argv, then override lef/def discovered from dataset
    Parameters parameters(argc, argv);
    if (!lef_path.empty()) parameters.lef_file = lef_path;
    if (!def_path.empty()) parameters.def_file = def_path;
    Design design(parameters); // constructor calls read(lef, def)

    // derive design_name from def filename if dataset not used
    std::string design_name;
    if (!dataset.empty()) design_name = dataset;
    else {
        auto pos = def_path.find_last_of("\\/");
        std::string fname = (pos==std::string::npos) ? def_path : def_path.substr(pos+1);
        auto dot = fname.find_last_of('.');
        design_name = (dot==std::string::npos) ? fname : fname.substr(0, dot);
    }

    std::string out_dir = out_root + "/" + design_name;
    design.dump_net_terminals_by_degree(out_dir, max_degree, normalize);

    std::cout << "Dump finished. Exiting." << std::endl;
    return 0;
}