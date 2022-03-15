# Contributing
KMK is a community effort and welcomes contributions of code and documentation from people 
of all backgrounds and levels of technical skill. As such, these guidelines should serve 
to make contributing as easy as possible for everyone while maintaining a consistent style.

## Contributing Code
The following guidelines should ensure that any code contributed can be merged in as 
painlessly as possible. If you're unsure how to set up your development environment, 
feel free to join the chat, [#kmkfw:klar.sh on Matrix](https://matrix.to/#/#kmkfw:klar.sh). 
This channel is bridged to Discord [here](https://discord.gg/QBHUUpeGUd) for convenience.

### Code Style

KMK uses [Black](https://github.com/psf/black) with a Python 3.6 target and,
[(controversially?)](https://github.com/psf/black/issues/594) single quotes.
Further code styling is enforced with isort and flake8 with several plugins.
`make fix-isort fix-formatting` before a commit is a good idea, and CI will fail
if inbound code does not adhere to these formatting rules. Some exceptions are
found in `setup.cfg` loosening the rules in isolated cases, notably
`user_keymaps` (which is *also* not subject to Black formatting for reasons
documented in `pyproject.toml`).

### Tests

Unit tests within the `tests` folder mock various CicuitPython modules to allow
them to be executed in a desktop development environment.

Execute tests using the command `python -m unittest`.

## Contriburing Documentation
While KMK welcomes documentation from anyone with and understanding of the issues 
and a willingness to write them up, it's a good idea to familiarize yourself with 
the docs. Documentation should be informative but concise.

### Styling
Docs are written and rendered in GitHub Markdown. A comprehensive guide to GitHub's 
Markdown can be found [here](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

In particular, KMK's docs should include a title, demarcated with `#`, and subheadings 
should be demarcated with `##`, `###`, and so on. Headings should be short and specific.

### Example Code
Where possible, practical code examples should be included in documentation to help 
new users understand how to implement features. In general, it's better to have a code-
block with comments inside it rather than multiple blocks of code broken up with 
explanation.

Code blocks should be formatted as Python code as follows:
````
```python
print('Hello, world!')
```
````

Inline code, indicated with ``backticks``, should be used when calling out specific 
functions or Python keywords within the body of paragraphs or in lists.