import typer

app = typer.Typer()  # Initialize Typer application

@app.command()
def hello():
    """Simple test command that prints a greeting."""
    print("Hello, world!")

@app.command()
def version():
    """Show the package version."""
    # In a real project, you might import __version__ from your package
    print("ibd-quantum-esm version 0.1.0")
