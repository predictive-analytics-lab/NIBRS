import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    commands = []

    db_name = args.db_name

    commands.append(f"createdb {db_name}")

    dl_path = (Path(__file__).parent / 'downloads').resolve()
    for data_dir in dl_path.iterdir():
        if next(data_dir.iterdir()).is_dir(): # Sometimes /STATE-YEAR/ containts /STATE/
            data_dir = next(data_dir.iterdir())
        if (data_dir / 'postgres_setup.sql').is_file():
            commands.append(f"cd {data_dir.resolve()}")
            commands.append(f"psql {db_name} < postgres_setup.sql")
            break
    for data_dir in dl_path.iterdir():
        if next(data_dir.iterdir()).is_dir(): # Sometimes /STATE-YEAR/ containts /STATE/
            data_dir = next(data_dir.iterdir())
        if (data_dir / 'postgres_load.sql').is_file():
            commands.append(f"cd {data_dir.resolve()}")
            commands.append(f"psql {db_name} < postgres_load.sql")
        else:
            print(f"{str(data_dir / 'postgres_load.sql')} missing")

    if args.dry_run:
        print("\n".join(commands))
    else:
        commands_path = Path(__file__).parent / f"create_{db_name}.sh"
        commands_path.write_text("\n".join(commands))


if __name__ == "__main__":
    main()
