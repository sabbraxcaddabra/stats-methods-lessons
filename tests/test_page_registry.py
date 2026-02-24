import unittest

from page_registry import get_content_pages, split_content_pages


class PageRegistryTests(unittest.TestCase):
    def test_content_pages_have_human_titles_and_expected_paths(self) -> None:
        expected = {
            "lesson_1": ("Урок 1: Байес и минимакс", "/lesson_1"),
            "lesson_2": ("Урок 2: Модель пробития", "/lesson_2"),
            "lesson_3": ("Урок 3: Случайные процессы", "/lesson_3"),
            "linear_regression_from_scratch": (
                "Линейная регрессия с нуля",
                "/linear-regression",
            ),
            "gp_from_scratch": (
                "Регрессия на гауссовских процессах с нуля",
                "/gauss-process-regression",
            ),
        }
        pages = get_content_pages()

        self.assertEqual({page.name for page in pages}, set(expected))
        for page in pages:
            title, path = expected[page.name]
            self.assertEqual(page.title, title)
            self.assertEqual(page.path, path)

    def test_split_content_pages_groups_lessons_and_extra(self) -> None:
        pages = get_content_pages()
        lessons, extra = split_content_pages(pages)

        self.assertEqual(
            [page.name for page in lessons],
            ["lesson_1", "lesson_2", "lesson_3"],
        )
        self.assertEqual(
            [page.name for page in extra],
            ["linear_regression_from_scratch", "gp_from_scratch"],
        )


if __name__ == "__main__":
    unittest.main()
