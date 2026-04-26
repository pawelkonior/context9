from __future__ import annotations

from typing import TYPE_CHECKING

from context9 import main as cli_main

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_main_delegates_to_ingestion_flow(mocker: MockerFixture) -> None:
    run = mocker.patch("context9.main.run")

    assert cli_main() is None
    run.assert_called_once_with()
