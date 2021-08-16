import argparse
from pathlib import Path

def get_year(path: Path) -> str:
    if "-" in path.name:
        return path.name.split("-")[-1]
    else:
        return path.parent.name.split("-")[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    commands = []

    db_name = args.db_name
    


    dl_path = (Path(__file__).parent / 'downloads').resolve()
    
    years = list({get_year(data_dir) for data_dir in dl_path.iterdir()})
    
    for year in years:
        commands.append(f"createdb {db_name}_{year}")

    for data_dir in dl_path.iterdir():
        if next(data_dir.iterdir()).is_dir(): # Sometimes /STATE-YEAR/ containts /STATE/
            data_dir = next(data_dir.iterdir())
        if (data_dir / 'postgres_setup.sql').is_file():
            commands.append(f"cd {data_dir.resolve()}")
            for year in years:
                commands.append(f"psql {db_name}_{year} < postgres_setup.sql")
            break
    for data_dir in dl_path.iterdir():
        if next(data_dir.iterdir()).is_dir(): # Sometimes /STATE-YEAR/ containts /STATE/
            data_dir = next(data_dir.iterdir())
        if (data_dir / 'postgres_load.sql').is_file():
            commands.append(f"cd {data_dir.resolve()}")
            commands.append(f"psql {db_name}_{get_year(data_dir)} < postgres_load.sql")
        else:
            print(f"{str(data_dir / 'postgres_load.sql')} missing")

    if args.dry_run:
        print("\n".join(commands))
    else:
        commands_path = Path(__file__).parent / f"create_{db_name}.sh"
        commands_path.write_text("\n".join(commands))


if __name__ == "__main__":
    main()
