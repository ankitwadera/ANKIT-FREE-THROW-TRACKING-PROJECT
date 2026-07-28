from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


APP_URL = "http://localhost:8501"
OUTPUT_FOLDER = Path("docs/screenshots")


async def main() -> None:
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True
        )

        page = await browser.new_page(
            viewport={
                "width": 1600,
                "height": 1000,
            },
            device_scale_factor=1,
        )

        await page.goto(
            APP_URL,
            wait_until="networkidle",
        )

        await page.screenshot(
            path=str(
                OUTPUT_FOLDER
                / "01-home.png"
            ),
            full_page=True,
        )

        # Public mode can be enabled with APP_MODE=public before launching.
        # The remaining pages should be captured manually after selecting:
        # Executive Demo, Session Analysis, Player Profile, Team Dashboard,
        # and Reports Center.
        print(
            "Captured docs/screenshots/01-home.png"
        )

        await browser.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
