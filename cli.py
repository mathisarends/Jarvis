import click
import time


def display_jarvis_logo():
    """Display the epic JARVIS logo with colors and effects"""

    # Header with glowing effect
    click.echo(
        click.style("┌─[ ", fg="cyan", bold=True)
        + click.style("SYSTEM INITIALIZING", fg="bright_green", bold=True, blink=True)
        + click.style(" ]─────────────────────────────────┐", fg="cyan", bold=True)
    )

    click.echo(
        click.style("│", fg="cyan", bold=True)
        + " " * 63
        + click.style("│", fg="cyan", bold=True)
    )

    # JARVIS ASCII with gradient-like colors
    jarvis_lines = [
        "       ██╗  █████╗ ██████╗ ██╗   ██╗██╗███████╗",
        "       ██║ ██╔══██╗██╔══██╗██║   ██║██║██╔════╝",
        "       ██║ ███████║██████╔╝██║   ██║██║███████╗",
        "  ██   ██║ ██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║",
        "  ╚█████╔╝ ██║  ██║██║  ██║ ╚████╔╝ ██║███████║",
        "   ╚════╝  ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝",
    ]

    colors = [
        "bright_red",
        "bright_yellow",
        "bright_green",
        "bright_cyan",
        "bright_blue",
        "bright_magenta",
    ]

    for i, line in enumerate(jarvis_lines):
        click.echo(
            click.style("│", fg="cyan", bold=True)
            + " " * 8
            + click.style(line, fg=colors[i], bold=True)
            + " " * 8
            + click.style("│", fg="cyan", bold=True)
        )
        time.sleep(0.1)

    click.echo(
        click.style("│", fg="cyan", bold=True)
        + " " * 63
        + click.style("│", fg="cyan", bold=True)
    )

    # Subtitle with fire effect
    subtitle = "🔥 Just A Rather Very Intelligent System 🔥"
    padding = (63 - len(subtitle)) // 2
    click.echo(
        click.style("│", fg="cyan", bold=True)
        + " " * padding
        + click.style(subtitle, fg="bright_red", bold=True)
        + " " * (63 - padding - len(subtitle))
        + click.style("│", fg="cyan", bold=True)
    )

    click.echo(
        click.style("│", fg="cyan", bold=True)
        + " " * 63
        + click.style("│", fg="cyan", bold=True)
    )

    # Footer with status
    click.echo(
        click.style("└─[ ", fg="cyan", bold=True)
        + click.style("STATUS: ONLINE", fg="bright_green", bold=True, blink=True)
        + click.style(" ]─────────────────────────────────────┘", fg="cyan", bold=True)
    )


def display_welcome_message():
    """Display welcome message"""
    click.echo()
    click.echo(
        click.style(
            "🎯 JARVIS SYSTEM FULLY OPERATIONAL 🎯", fg="bright_green", bold=True
        )
    )
    click.echo(
        click.style("   Ready to assist you, Sir.", fg="bright_cyan", italic=True)
    )
    click.echo()


@click.command()
@click.option("--fast", "-f", is_flag=True, help="Skip animations for faster loading")
def main(fast):
    """🤖 JARVIS - Just A Rather Very Intelligent System"""

    # Clear screen for dramatic effect
    click.clear()

    # Display the epic logo
    display_jarvis_logo()

    # Simple welcome message
    display_welcome_message()


if __name__ == "__main__":

    main()
