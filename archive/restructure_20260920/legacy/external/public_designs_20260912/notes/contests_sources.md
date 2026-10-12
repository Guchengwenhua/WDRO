# Public contest benchmark acquisition, 2026-09-12 UTC

All acquired files, helper scripts, and evidence are below `DRO/Codex/benchmarks/public_designs_20260912/`. No frozen project code, model, GPU job, or configuration was changed by this acquisition task.

## Downloaded and parsed: ISPD 2008 Global Routing, all 16 designs

Official source and input specification: https://www.ispd.cc/contests/08/ispd08rc.html

The organizers describe these as industrial ASIC derived routing instances. The ASCII input supplies each original net's pin locations as physical XY and metal layer, with grid/capacity metadata. These are public physical-design benchmarks, not complete fabricated-chip manufacturing data or newly synthesized netlists. For this project's 2-D REST interface, `parsers/contests.py:parse_design(path)` yields `(net_id, np.ndarray[n,2])` and drops only the layer column. It preserves raw degree, pin order, physical XY, and every repeated coordinate. It never rounds coordinates to routing tiles and never deduplicates/reassigns a degree. The parent evaluation applies the frozen normalization and expected-degree validity rules.

All 16 official URLs returned HTTP 200 with gzip content on 2026-09-12. Downloads used 6 threads, a 3 GB per compressed/expanded file guard, content-length checks, gzip validation, and `.partial` followed by atomic publication. No downloaded code was executed. The archive and expanded-file SHA256 hashes, sizes, full source URLs, parser name, and original family labels are in `sources/contests/designs_manifest.json`. Archive bytes total **354,471,191**; expanded input bytes total **1,107,023,148**. All 16 expanded-file SHA256 hashes differ.

Data license: the official page publicly provides the academic competition inputs; it does not give a separate explicit dataset license. Its “openly available tools” conditions concern contestant software, not a blanket data license. We claim no unrestricted redistribution license. Local research evaluation uses the publicly supplied inputs with attribution.

Deduplication/lineage: 8 cases in this suite are the ISPD 2007 3-D contest cases reused in ISPD 2008; the official HTML repeats its adaptec1 download link, which was deduplicated by URL. We downloaded each of the 16 distinct cases exactly once. Adaptec/bigblue/newblue names are related to prior ISPD 2005/2006 placement families; do not treat different contest versions or placements as unrelated original chip designs. These 16 do not share design names with the project's ISPD 2018/2019 routing inputs, but cross-dataset geometric duplicate checks belong to the evaluation extraction stage.

## Independent coordinate audit

`sources/contests/coordinate_audit.json` performed a complete strict read of **13,702,396 nets / 53,059,118 pins**, checked unique net identifiers, finite XY, valid layers, coordinate extrema, every raw degree, and repeated XY on target degrees. The parser's output agrees with an independent manual read of the official first net; gzip/plain streams agree for the first 100 nets. A test with pins on two different layers at the same XY preserves all three original pins. Truncated input, invalid layers, and nonfinite XY are rejected. Test evidence: `sources/contests/parser_contract_verification.json`; scripts: `audit_contests.py`, `test_parser_contract.py`.

Raw target-degree totals: degree 10 = 97,283; 20 = 15,706; 30 = 9,096; 40 = 5,441; 50 = 3,701; total = **131,227**. Of these, 18 nets have repeated XY (5 at degree 10, 5 at 20, 8 at 30), and zero target-degree nets have zero span. The frozen spacing rule may reject additional nonduplicate nets; these raw counts are not final evaluation acceptance counts.

| Design | Original nets | d10 | d20 | d30 | d40 | d50 | Repeated XY at target degrees |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptec1 | 219794 | 1867 | 306 | 54 | 173 | 106 | 0 |
| adaptec2 | 260159 | 2209 | 368 | 146 | 126 | 122 | 0 |
| adaptec3 | 466295 | 3130 | 616 | 364 | 311 | 128 | 0 |
| adaptec4 | 515304 | 2547 | 467 | 234 | 285 | 255 | 0 |
| adaptec5 | 867441 | 7730 | 995 | 320 | 311 | 470 | 0 |
| newblue1 | 331663 | 2453 | 299 | 41 | 94 | 84 | 0 |
| newblue2 | 463213 | 3696 | 369 | 344 | 223 | 200 | 0 |
| newblue3 | 551667 | 2334 | 256 | 251 | 162 | 122 | 18 |
| bigblue1 | 282974 | 2318 | 295 | 220 | 180 | 11 | 0 |
| bigblue2 | 576816 | 4325 | 496 | 311 | 250 | 63 | 0 |
| bigblue3 | 1122340 | 6134 | 848 | 1110 | 472 | 81 | 0 |
| bigblue4 | 2228903 | 19290 | 2200 | 959 | 697 | 636 | 0 |
| newblue4 | 636195 | 4834 | 323 | 220 | 160 | 150 | 0 |
| newblue5 | 1257555 | 7356 | 1579 | 1320 | 538 | 257 | 0 |
| newblue6 | 1286452 | 9911 | 2301 | 851 | 563 | 437 | 0 |
| newblue7 | 2635625 | 17149 | 3988 | 2351 | 896 | 579 | 0 |

## Researched alternatives, not included in this evaluation acquisition

- **ICCAD 2015 incremental timing-driven placement:** https://www.iccad-contest.org/2015/problem_C/default.html . The source page now lists all 8 superblue cases (including 3 originally hidden final cases). The organizers state the cases are reverse engineered from earlier Bookshelf data, so their logic/timing can differ from the original real designs. Every linked official `Benchmarks/superblue{1,3,4,5,7,10,16,18}.tar.gz` URL returned HTTP 404 during this run. `iccad2015_urls_head.json` preserves failures. Format PDF remains reachable at https://www.iccad-contest.org/2015/problem_C/Docs/contest_file_formats.pdf . These cases would require validated LEF/DEF/Verilog pin reconstruction.
- **ICCAD 2017 Multi-Deck Standard Cell Legalization:** https://www.iccad-contest.org/2017/Problem_C/default.html . The source describes LEF/DEF physical placements derived from ISPD 2015, with global placements produced by Eh?Placer. Linked sample/benchmark/hidden tarballs currently return HTTP 404; evidence `iccad2017_urls_head.json`. The page also contains an organizer-provided historical Google Drive link https://drive.google.com/open?id=0B8sY-XEqkT6BdlRhellIZTJ4bk0 ; this fallback was not downloaded or accessibility-validated. Cross-family lineage with ISPD 2014/2015 must be retained.
- **Author mirror of ICCAD 2015:** the official DATE 2025 Efficient-TDP implementation https://github.com/lamda-bbo/Efficient-TDP links https://drive.google.com/file/d/1HsAW_qcRje_-Ex1anWqAEQOKpGeCxpZa/view?usp=sharing for its 8-case `iccad2015.hs` package. This is an author-published adapted benchmark package; its timing collateral may differ from the original. Download size and file content not validated in this run.
- **Second author mirror:** the official BBOPlace-Bench implementation https://github.com/lamda-bbo/BBOPlace-Bench links https://drive.google.com/file/d/1JEC17FmL2cM8BEAewENvRyG6aWxH53mX/view?usp=sharing . Browser lookup identifies `iccad2015.ot.zip`. It includes the same 8 superblue names. This is an author source rather than the original organizer; no checksum/content equivalence or size claim is made. It was not downloaded this round.
- **ISPD 2005 Bookshelf originals and submitted placements:** https://www.ispd.cc/contests/05/contest.htm . Source provides 8 benchmark archives and multiple teams' placement solutions, but underlying adaptec/bigblue families overlap the acquired ISPD07/08 routing family. Adding them without a lineage split would overstate dataset diversity; therefore no redundant acquisition was performed.

The full raw source HTML pages for ISPD08, ICCAD15, and ICCAD17 are retained beside these records. Public web searches located all sources before shell-side network retrieval.
