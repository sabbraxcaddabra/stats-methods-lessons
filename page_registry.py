from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PageSpec:
    name: str
    title: str
    path: str
    root: str


PAGE_SPECS: tuple[PageSpec, ...] = (
    PageSpec(
        name="index",
        title="Главная",
        path="",
        root="./pages/index.py",
    ),
    PageSpec(
        name="lesson_1",
        title="Лабораторная работа #1: Статистические методы принятия решений",
        path="/lesson_1",
        root="./pages/lesson_1.py",
    ),
    PageSpec(
        name="lesson_2",
        title="Лабораторная работа #2: Проверка статистических гипотез",
        path="/lesson_2",
        root="./pages/lesson_2.py",
    ),
    PageSpec(
        name="lesson_3",
        title="Лабораторная работа #3: Случайные процессы",
        path="/lesson_3",
        root="./pages/lesson_3.py",
    ),
    PageSpec(
        name="linear_regression_from_scratch",
        title="Линейная регрессия с нуля",
        path="/linear-regression",
        root="./pages/linear_regression_from_scratch.py",
    ),
    PageSpec(
        name="gp_from_scratch",
        title="Регрессия на гауссовских процессах с нуля",
        path="/gauss-process-regression",
        root="./pages/gp_from_scratch.py",
    ),
)


def get_content_pages() -> list[PageSpec]:
    return [page for page in PAGE_SPECS if page.path]


def split_content_pages(pages: list[PageSpec]) -> tuple[list[PageSpec], list[PageSpec]]:
    lesson_pages = [page for page in pages if page.name.startswith("lesson_")]
    lesson_pages.sort(key=_lesson_sort_key)

    extra_pages = [page for page in pages if not page.name.startswith("lesson_")]
    extra_pages.sort(key=lambda page: page.title.lower())

    return lesson_pages, extra_pages


def _lesson_sort_key(page: PageSpec) -> tuple[int, str]:
    match = re.match(r"lesson_(\d+)$", page.name)
    if match is None:
        return (10**9, page.name)
    return (int(match.group(1)), page.name)
