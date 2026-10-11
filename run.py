"""ReplyDesk entry point.

    python run.py scrape --query "restaurants Baltimore Avenue Philadelphia" --max 10 [--from-cache]
    python run.py list
    python run.py migrate

Project modules are imported inside each command so one missing module never breaks the others.
"""
import argparse
import sys


def cmd_scrape(args):
    import scrape

    businesses = scrape.scrape(args.query, args.max, use_cache=args.from_cache)
    scrape.print_table(businesses, scrape.LAST_RUN)


def cmd_list(args):
    import scrape
    import store

    scrape.print_table(store.list_businesses())


def cmd_migrate(args):
    import store

    counts = store.migrate_json_to_mongo()
    for key, value in counts.items():
        print(f"{key}: {value}")


def cmd_generate(args):
    print("generate: coming in a later checkpoint")


def cmd_serve(args):
    print("serve: coming in a later checkpoint")


def cmd_outreach(args):
    print("outreach: coming in a later checkpoint")


def build_parser():
    parser = argparse.ArgumentParser(prog="run.py", description="ReplyDesk: pre-written Google review replies")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scrape", help="find businesses and their newest reviews (Apify)")
    p.add_argument("--query", required=True, help='e.g. "restaurants Baltimore Avenue Philadelphia"')
    p.add_argument("--max", type=int, default=10, help="max places to discover (default 10)")
    p.add_argument("--from-cache", action="store_true", help="reuse data/raw results, no Apify calls")
    p.set_defaults(func=cmd_scrape)

    sub.add_parser("list", help="table of stored businesses").set_defaults(func=cmd_list)
    sub.add_parser("migrate", help="copy the JSON store into MongoDB").set_defaults(func=cmd_migrate)
    sub.add_parser("generate", help="write replies (later checkpoint)").set_defaults(func=cmd_generate)
    sub.add_parser("serve", help="preview web app (later checkpoint)").set_defaults(func=cmd_serve)
    sub.add_parser("outreach", help="owner outreach (later checkpoint)").set_defaults(func=cmd_outreach)
    return parser


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # a caller capturing output may have swapped in a StringIO
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, OSError) as exc:  # OSError: e.g. migrate with no JSON store, store lock busy
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
