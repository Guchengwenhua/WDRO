"""Run the frozen stage-6 smoke with new stage-5 comparison output isolated."""
import run_stage5_smoke as stage5_smoke
import run_stage6_smoke as stage6_smoke


def main():
    previous_log = stage5_smoke.LOG
    stage5_smoke.LOG = stage6_smoke.LOG / 'stage5_selected_equivalence'
    try:
        return stage6_smoke.main()
    finally:
        stage5_smoke.LOG = previous_log


if __name__ == '__main__':
    main()
