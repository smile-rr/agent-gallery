from time import sleep

from rich.console import Console
from rich import print as rprint
from rich.progress import track
from rich.table import Table
from rich.spinner import Spinner
from rich.pretty import Pretty, install

install()
console = Console()

test_data = [
    {"jsonrpc": "2.0", "method": "sum", "params": [None, 1, 2, 4, False, True], "id": "1",},
    {"jsonrpc": "2.0", "method": "notify_hello", "params": [7]},
    {"jsonrpc": "2.0", "method": "subtract", "params": [42, 23], "id": "2"},
]

def test_log():
    enabled = False
    context = {
        "foo": "bar",
    }
    movies = ["Deadpool", "Rise of the Skywalker"]
    console.log("Hello from", console, "!")
    console.log(test_data, log_locals=True)
    console.print(":smiley: :vampire: :pile_of_poo: :thumbs_up: :raccoon:")

def test_log_json():
    rprint("Hello World")
    rprint(test_data)

def test_table():
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", style="dim", width=12)
    table.add_column("Title")
    table.add_column("Production Budget", justify="right")
    table.add_column("Box Office", justify="right")
    table.add_row(
        "Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
    )
    table.add_row(
        "May 25, 2018",
        "[red]Solo[/red]: A Star Wars Story",
        "$275,000,000",
        "$393,151,347",
    )
    table.add_row(
        "Dec 15, 2017",
        "Star Wars Ep. VIII: The Last Jedi",
        "$262,000,000",
        "[bold]$1,332,539,889[/bold]",
    )

    console.print(table)


def do_step(step):
    pass

def test_progress_bar():
    for step in track(range(10)):
        sleep(0.1)
        do_step(step)
    for step in track(range(10)):
        sleep(0.1)
        do_step(step)
def test_spinner():
    tasks = [f"task {n}" for n in range(1, 11)]
    spinner = Spinner("shark")
    with console.status(spinner) as status:
        while tasks:
            task = tasks.pop(0)
            sleep(1)
            console.log(f"{task} complete")


def test_columns():
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", style="dim", width=12)
    table.add_column()



if __name__ == "__main__":
    test_log()
    rprint("="*40)
    test_log_json()

    rprint("="*40)
    test_table()

    rprint("="*40)
    test_progress_bar()

    rprint("="*40)
    test_spinner()

    rprint("="*40)
    test_columns()