"""Tadpole launcher: `python main.py [trace.dwc]`."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from tadpole.app.main import main
    main()
