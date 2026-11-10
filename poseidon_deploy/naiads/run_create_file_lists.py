"""Split a folder of images into file lists for an HPC job array.

Writes ``file_list_1.txt`` ... ``file_list_<N>.txt``, one per array
task; run before segmentation (stage 02) and label generation (03).

Example
-------
``python run_create_file_lists.py --image_dir images/daylight
--num_jobs 32``
"""

from poseidon_utils.file_organizer import prepare_job_lists

import cli_common


def main(argv=None):
    """
    Write job array file lists.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    parser.add_argument("--image_dir", required=True,
                        help="Folder whose files are split into lists.")
    parser.add_argument("--num_jobs", type=int, required=True,
                        help="Number of job array tasks (file lists).")
    parser.add_argument("--output_dir", default=None,
                        help="Folder for the lists; default is "
                             "'job_file_lists' beside --image_dir.")
    parser.add_argument("--skip_done_dir", default=None,
                        help="Leave out images whose <stem>_predseg.png "
                             "is already in this folder (the predictions "
                             "folder), to list only what is left.")
    args = cli_common.start(parser, argv)
    lists = prepare_job_lists(args.image_dir, args.num_jobs, args.output_dir,
                              skip_done_dir=args.skip_done_dir)
    print(f"Wrote {len(lists)} file lists; submit with --array=1-{len(lists)}")


if __name__ == "__main__":
    main()
