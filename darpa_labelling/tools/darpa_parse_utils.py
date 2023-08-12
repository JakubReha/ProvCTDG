from datetime import datetime
import pytz
import argparse


class EdgeType_t:
    Vfork = 1
    Clone = 2
    Execve = 3
    Kill = 4
    Pipe = 5
    Delete = 6
    Create = 7
    Recv = 8
    Send = 9
    Mkdir = 10
    Rmdir = 11
    Open = 12
    Load = 13
    Read = 14
    Write = 15
    Connect = 16
    Getpeername = 17
    Filepath = 18
    Mode = 19
    Mtime = 20
    Linknum = 21
    Uid = 22
    Count = 23
    Nametype = 24
    Version = 25
    Dev = 26
    SizeByte = 27
    EdgeType_NR_ITEMS = 28
    NotDefined = 0


def get_edge_type(num):
    for name, value in vars(EdgeType_t).items():
        if value == num:
            return name
    return "NotDefined"


def to_tz(dt: datetime, tz: str = "UTC") -> datetime:
    tzo = pytz.timezone(tz)
    tz_datetime = tzo.localize(dt)  # dt.replace(tzinfo=tzo)
    return tz_datetime


def nanoseconds_to_datetime(nanoseconds):
    # Convert nanoseconds to seconds
    seconds = int(nanoseconds / 1_000_000_000)

    # Create a datetime object from the seconds
    dt = datetime.utcfromtimestamp(seconds)
    utc_datetime = to_tz(dt)

    # Convert the datetime object to the GMT-4 timezone
    darpa_tz = pytz.timezone("US/Eastern")
    darpa_datetime = utc_datetime.astimezone(darpa_tz)

    return darpa_datetime


def valid_date(dt):
    if not isinstance(dt, datetime):
        try:
            dt = datetime.strptime(dt, "%Y-%m-%d_%H:%M")
        except:
            raise ValueError("No valid date format found.")
    return dt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp",
        "-t",
        nargs="?",
        default=None,
        type=int,
        help="Nanosecond timestamp",
    )
    parser.add_argument(
        "--date",
        "-d",
        type=valid_date,
        nargs="?",
        default=None,
        help="Format e.g., 2022-03-04_09:46, DARPA timezone",
    )
    parser.add_argument(
        "--edgeid", "-e", nargs="?", default=None, type=int, help="System call integer"
    )

    args = parser.parse_args()

    ts = args.timestamp
    if args.timestamp is not None:
        print(
            f"Occurrence in darpa event log format: {nanoseconds_to_datetime(args.timestamp)}"
        )

    if args.date is not None:
        dt = to_tz(args.date, "US/Eastern")
        dt = dt.astimezone(pytz.timezone("UTC"))
        dt = int(dt.timestamp() * 1e9)
        print(f"Event log date to timestamp in nanoseconds: {dt}")

    edgeid = args.edgeid
    if edgeid is not None:
        print(f"{edgeid=} corresponds to: {get_edge_type(edgeid)}")
