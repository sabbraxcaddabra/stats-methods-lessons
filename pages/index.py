import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(get_content_pages, mo, split_content_pages):
    def as_markdown_links(items: list[tuple[str, str]]) -> str:
        if not items:
            return "- Пока пусто"
        return "\n".join(f"- [{title}]({path})" for title, path in items)

    pages = get_content_pages()
    lesson_pages, extra_pages = split_content_pages(pages)

    lessons_md = as_markdown_links([(page.title, page.path) for page in lesson_pages])
    extra_md = as_markdown_links([(page.title, page.path) for page in extra_pages])

    mo.vstack(
        [
            mo.md("# Статистические методы"),
            mo.md("Выберите страницу:"),
            mo.md(f"## Уроки\n{lessons_md}"),
            mo.md(f"## Дополнительно\n{extra_md}"),
        ]
    )
    return


@app.cell
def __():
    import marimo as mo
    from page_registry import get_content_pages, split_content_pages

    return mo, get_content_pages, split_content_pages


if __name__ == "__main__":
    app.run()
