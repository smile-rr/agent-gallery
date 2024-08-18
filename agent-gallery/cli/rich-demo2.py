import sys
import time
from rich import print
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import emoji
from rich.progress import track
from rich.spinner import Spinner
from rich.panel import Panel

# Initialize the console
console = Console()

# Function to display a colorful greeting
def display_greeting(name):
    console.print(f"[bold green]Hello, {name}![/bold green] Welcome to Happy!")
    console.print(f"[bold green]Let's have some fun! 😊[/bold green]")

# Function to plot a sample chart
def plot_sample_chart():
    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [10, 15, 25, 30, 35]

    # Plotting the chart
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='blue')
    plt.title("Sample Chart")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

# Function to display a table of options
def display_options():
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option")
    table.add_column("Description")
    table.add_row("1", "Display Greeting")
    table.add_row("2", "Plot Sample Chart")
    table.add_row("3", "Shining Smile")
    table.add_row("4", "Progress Bar")
    table.add_row("5", "Spinner")
    table.add_row("6", "Panel")
    table.add_row("7", "Exit")

    console.print(table)

# Function to display a shining smile
def display_shining_smile():
    console.print("[bold green]✨ Shining Smile ✨[/bold green]")
    console.print(emoji.emojize(":grinning_face_with_big_eyes:"))

# Function to display a progress bar
def display_progress_bar():
    console.print("[bold green]Progress Bar:")
    for i in track(range(100), description="[green]Processing..."):
        pass

# Function to display a spinner
def display_spinner():
    with Spinner("dots") as task:
        # Simulate loading
        for _ in range(10):
            sys.stdout.flush()
            sys.stdout.write('.')
            sys.stdout.flush()
            sys.stdout.write('\b')
            sys.stdout.flush()
            # Update the spinner text
            console.log("[progress.description]{task.fields[description]}".format(task=task))
            task.update(description="Loading...")
            time.sleep(0.5)

# Function to display a panel
def display_panel():
    console.print(Panel.fit("Welcome to the Panel!", title="[bold green]Panel Example[/bold green]", border_style="green"))

# Main function
def main():
    name = Prompt.ask("[bold green]Please enter your name:")
    display_greeting(name)

    while True:
        display_options()
        choice = Prompt.ask("[bold green]Choose an option (1-7):", choices=["1", "2", "3", "4", "5", "6", "7"])

        if choice == "1":
            display_greeting(name)
        elif choice == "2":
            plot_sample_chart()
        elif choice == "3":
            display_shining_smile()
        elif choice == "4":
            display_progress_bar()
        elif choice == "5":
            display_spinner()
        elif choice == "6":
            display_panel()
        elif choice == "7":
            console.print("[bold red]Exiting the program.[/bold red]")
            sys.exit(0)

if __name__ == "__main__":
    main()