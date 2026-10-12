"""ReplyDesk entry point.

    python run.py scrape --query "restaurants Baltimore Avenue Philadelphia" --max 10 [--from-cache]
    python run.py list
    python run.py generate --business <slug> [--redo] [--samples N]   or   --all [--redo]
    python run.py samples --business <slug> [--n N]
    python run.py serve [--reload]                       (0.0.0.0:8000)
    python run.py migrate                                (JSON store -> MongoDB)
    python run.py outreach                               (data/outreach.csv)

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
    import generate

    generate.generate(None if args.all else args.business, redo=args.redo)
    if args.business and args.samples:
        generate.print_samples(args.business, args.samples)


def cmd_samples(args):
    import generate

    generate.print_samples(args.business, args.n)


def cmd_serve(args):
    import uvicorn

    import config

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=args.reload,
                reload_dirs=[str(config.BASE_DIR)] if args.reload else None)


def cmd_outreach(args):
    import outreach

    path = outreach.write_csv()
    rows = outreach.rows()
    print(f"{len(rows)} businesses -> {path}")
    for r in rows[:10]:
        print(f"  {r['unanswered']:>3} unanswered  {r['phone'] or '-':16} {r['status']:7} {r['name']}")


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
    p = sub.add_parser("generate", help="write replies for unanswered reviews (Luminal)")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--business", help="slug from run.py list")
    target.add_argument("--all", action="store_true", help="every business")
    p.add_argument("--redo", action="store_true", help="rewrite replies that already exist")
    p.add_argument("--samples", type=int, default=5, help="with --business: print N replies (default 5)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("samples", help="print replies next to their reviews")
    p.add_argument("--business", required=True, help="slug from run.py list")
    p.add_argument("--n", type=int, default=5, help="how many (default 5)")
    p.set_defaults(func=cmd_samples)
    p = sub.add_parser("serve", help="web app on 0.0.0.0:8000 (landing page, owner preview and full set, "
                                     "Reply Studio, admin)")
    p.add_argument("--reload", action="store_true", help="restart automatically when code changes")
    p.set_defaults(func=cmd_serve)
    sub.add_parser("outreach", help="write data/outreach.csv, most unanswered first").set_defaults(func=cmd_outreach)
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
