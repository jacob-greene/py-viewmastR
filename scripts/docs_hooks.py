"""MkDocs hook: make links written for GitHub work on the documentation site.

``docs/index.md`` and ``docs/changelog.md`` hold one line, ``--8<-- "<file>"``; the hook puts the
root README.md or CHANGELOG.md there. Their links are relative to the repository root, so the
hook points them at the site pages, and points links to files that are not pages (scripts,
LICENSE) at GitHub.
"""

import re
from pathlib import Path

REPO = "https://github.com/jacob-greene/py-viewmastR/blob/main/"
# Root-relative target -> site-relative target, for pages in docs/
PAGES = {
    "docs/img/": "img/",
    "docs/r_comparison.md": "r_comparison.md",
    "docs/development.md": "development.md",
    "tutorials/README.md": "tutorials/README.md",
}
INCLUDE = re.compile(r'^--8<-- "([^"]+)"\s*$', re.M)
LINK = re.compile(r"\]\((?!https?://|#)([^)\s]+)\)")


def _fix(target: str) -> str:
    target = target.removeprefix("../")
    for old, new in PAGES.items():
        if target.startswith(old):
            return new + target[len(old):]
    if target.startswith(("scripts/", "LICENSE", "python/", "rust/", "tests/")):
        return REPO + target
    return target


def on_page_markdown(markdown, page, config, files):
    root = Path(config["config_file_path"]).parent
    markdown = INCLUDE.sub(lambda m: (root / m.group(1)).read_text(), markdown)
    if page.file.src_uri in ("index.md", "changelog.md", "r_comparison.md"):
        markdown = LINK.sub(lambda m: "](" + _fix(m.group(1)) + ")", markdown)
    return markdown
