"""Run the available stage's original regression suite in a real module."""
import argparse,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def leaves(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):yield from leaves(test)
        else:yield test

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-stage6',action='store_true');parser.add_argument('--through-stage4',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    suite=unittest.TestLoader().discover(str(ROOT/'tests'))
    tests=unittest.TestSuite(t for t in leaves(suite) if (args.include_stage6 or not t.__class__.__module__.endswith('test_stage6'))
        and (not args.through_stage4 or 'test_stage5' not in t.__class__.__module__))
    log=args.output or ROOT.parent/'experiments/logs/stage2/stage2_unit_tests_20260905.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as out:
        out.write('precision_v2 replay regression suite; filename retained for exporter schema compatibility\n')
        result=unittest.TextTestRunner(stream=out,verbosity=2).run(tests)
    print(dict(tests=result.testsRun,passed=result.wasSuccessful(),log=str(log)))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
