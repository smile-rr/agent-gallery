from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from rich.panel import Panel

console = Console()

text_content = "This is some plain text."
markdown_content = """
## This is a Markdown header
* This is a list item
"""

# Create a Text object for the plain text
text_element = Text(text_content)

# Create a Markdown object for the Markdown content
markdown_element = Markdown(markdown_content)

# Concatenate the plain text and Markdown content into a single Text object
combined_text = text_element + Text.from_ansi(str(markdown_element))

# Wrap the combined text in a Panel
panel = Panel(
    combined_text,
    title="Combined Text and Markdown",
    border_style="green",
    padding=1
)

# Print the panel
console.print(panel)