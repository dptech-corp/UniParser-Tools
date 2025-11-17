import json
import traceback
import uuid
from dataclasses import asdict, dataclass
from typing import List, Union

import requests
from PIL import Image

from uniparser_tools.common.constant import (
    FormatFlag,
    IntEnum,
    Language,
    OrderingMethod,
    ParseMode,
    ParseModeTextual,
    StatusFlag,
)
from uniparser_tools.utils.image import dump_image_base64_str


def int_enum_factory(items):
    return {k: int(v) if isinstance(v, IntEnum) else v for k, v in items}


@dataclass
class TriggerFileData:
    token: str
    lang: Language
    sync: bool
    textual: Union[ParseModeTextual, bool]
    table: Union[ParseMode, bool]
    molecule: Union[ParseMode, bool]
    chart: Union[ParseMode, bool]
    figure: Union[ParseMode, bool]
    expression: Union[ParseMode, bool]
    equation: Union[ParseMode, bool]
    pages: List[int]
    timeout: int
    ordering_method: OrderingMethod


@dataclass
class TriggerURLData:
    url: str
    token: str
    lang: Language
    sync: bool
    textual: Union[ParseModeTextual, bool]
    table: Union[ParseMode, bool]
    molecule: Union[ParseMode, bool]
    chart: Union[ParseMode, bool]
    figure: Union[ParseMode, bool]
    expression: Union[ParseMode, bool]
    equation: Union[ParseMode, bool]
    pages: List[int]
    timeout: int
    ordering_method: OrderingMethod
    proxy: str


@dataclass
class GetResultData:
    token: str
    content: bool
    objects: bool
    pages_dict: bool
    pages_tree: bool
    molecule_source: bool


@dataclass
class GetFormattedData:
    token: str
    content: bool
    objects: bool
    pages_dict: bool
    pages_tree: bool
    molecule_source: bool
    textual: FormatFlag
    table: FormatFlag
    molecule: FormatFlag
    chart: FormatFlag
    figure: FormatFlag
    expression: FormatFlag
    equation: FormatFlag


class UniParserClient:
    def __init__(self, user: str, host: str):
        assert user, "user name can not be empty"
        assert host.startswith("http"), "host must start with http"
        self.user = uuid.uuid5(uuid.NAMESPACE_DNS, user)
        self.host = host

    @property
    def trigger_file_endpoint(self):
        return f"{self.host}/trigger-file-async"

    @property
    def trigger_url_endpoint(self):
        return f"{self.host}/trigger-url-async"

    @property
    def trigger_snip_endpoint(self):
        return f"{self.host}/trigger-snip-async"

    @property
    def get_result_endpoint(self):
        return f"{self.host}/get-result"

    @property
    def get_formatted_endpoint(self):
        return f"{self.host}/get-formatted"

    def to_token(self, task_id: str):
        token = uuid.uuid5(self.user, task_id).hex
        # assert re.match(r"^[-\._?=&a-zA-Z0-9]{1,128}$", token), f"token: {token} contains illegal characters"
        return token

    def trigger_file(
        self,
        file_path: str,
        token: str = None,
        lang: Language = Language.Unknown,
        sync: bool = True,
        textual: Union[ParseModeTextual, bool] = ParseModeTextual.DigitalExported,
        table: Union[ParseMode, bool] = ParseMode.Disable,
        molecule: Union[ParseMode, bool] = ParseMode.Disable,
        chart: Union[ParseMode, bool] = ParseMode.Disable,
        figure: Union[ParseMode, bool] = ParseMode.Disable,
        expression: Union[ParseMode, bool] = ParseMode.Disable,
        equation: Union[ParseMode, bool] = ParseMode.Disable,
        pages: List[int] = None,
        timeout: int = 1800,
        ordering_method: OrderingMethod = OrderingMethod.GapTree,
    ):
        """
        sync: True=同步解析，该请求会在解析完成后才返回; False=异步解析，该请求会立即返回，解析结果需要通过GetResult接口获取
        """
        if not token:
            token = self.to_token(file_path)
        trigger_data = TriggerFileData(
            token=token,
            lang=lang,
            sync=sync,
            textual=textual,
            table=table,
            molecule=molecule,
            chart=chart,
            figure=figure,
            expression=expression,
            equation=equation,
            pages=pages,
            timeout=timeout,
            ordering_method=ordering_method,
        )

        try:
            files = {"file": open(file_path, "rb")}
            data = asdict(trigger_data, dict_factory=int_enum_factory)
            response = requests.post(self.trigger_file_endpoint, files=files, data=data)
        except Exception:
            return {
                "status": StatusFlag.Error,
                "token": token,
                "message": "trigger file failed",
                "description": traceback.format_exc(),
            }

        try:
            return response.json()
        except json.decoder.JSONDecodeError:
            return {"status": StatusFlag.Error, "token": token, "message": response.text}

    def trigger_snip(
        self,
        snip_path: str,
        token: str = None,
        lang: Language = Language.Unknown,
        sync: bool = True,
        textual: Union[ParseModeTextual, bool] = ParseModeTextual.DigitalExported,
        table: Union[ParseMode, bool] = ParseMode.Disable,
        molecule: Union[ParseMode, bool] = ParseMode.Disable,
        chart: Union[ParseMode, bool] = ParseMode.Disable,
        figure: Union[ParseMode, bool] = ParseMode.Disable,
        expression: Union[ParseMode, bool] = ParseMode.Disable,
        equation: Union[ParseMode, bool] = ParseMode.Disable,
        pages: List[int] = None,
        timeout: int = 1800,
        ordering_method: OrderingMethod = OrderingMethod.GapTree,
    ):
        if not token:
            token = self.to_token(snip_path)
        trigger_data = TriggerFileData(
            token=token,
            lang=lang,
            sync=sync,
            textual=textual,
            table=table,
            molecule=molecule,
            chart=chart,
            figure=figure,
            expression=expression,
            equation=equation,
            pages=pages,
            timeout=timeout,
            ordering_method=ordering_method,
        )

        try:
            img = dump_image_base64_str(Image.open(snip_path).convert("RGB"))
            data = {"img": img, **asdict(trigger_data, dict_factory=int_enum_factory)}
            result = requests.post(self.trigger_snip_endpoint, data=data)
        except Exception:
            return {
                "status": StatusFlag.Error,
                "token": token,
                "message": "trigger snip failed",
                "description": traceback.format_exc(),
            }
        try:
            return result.json()
        except json.decoder.JSONDecodeError:
            return {"status": StatusFlag.Error, "token": token, "message": result.text}

    def trigger_url(
        self,
        pdf_url: str,
        token: str = None,
        lang: Language = Language.Unknown,
        sync: bool = True,
        textual: Union[ParseModeTextual, bool] = ParseModeTextual.DigitalExported,
        table: Union[ParseMode, bool] = ParseMode.Disable,
        molecule: Union[ParseMode, bool] = ParseMode.Disable,
        chart: Union[ParseMode, bool] = ParseMode.Disable,
        figure: Union[ParseMode, bool] = ParseMode.Disable,
        expression: Union[ParseMode, bool] = ParseMode.Disable,
        equation: Union[ParseMode, bool] = ParseMode.Disable,
        pages: List[int] = None,
        timeout: int = 1800,
        ordering_method: OrderingMethod = OrderingMethod.GapTree,
        proxy: str = None,
    ):
        if not token:
            token = self.to_token(pdf_url)
        trigger_data = TriggerURLData(
            url=pdf_url,
            token=token,
            lang=lang,
            sync=sync,
            textual=textual,
            table=table,
            molecule=molecule,
            chart=chart,
            figure=figure,
            expression=expression,
            equation=equation,
            pages=pages,
            timeout=timeout,
            ordering_method=ordering_method,
            proxy=proxy,
        )
        try:
            result = requests.post(self.trigger_url_endpoint, json=asdict(trigger_data, dict_factory=int_enum_factory))
        except Exception:
            return {
                "status": StatusFlag.Error,
                "token": token,
                "message": "trigger url failed",
                "description": traceback.format_exc(),
            }
        try:
            return result.json()
        except json.decoder.JSONDecodeError:
            return {"status": StatusFlag.Error, "token": token, "message": result.text}

    def get_result(
        self,
        token: str,
        content: bool = False,
        objects: bool = False,
        pages_dict: bool = False,
        pages_tree: bool = False,
        molecule_source: bool = False,
    ):
        data = GetResultData(
            token=token,
            content=content,
            objects=objects,
            pages_dict=pages_dict,
            pages_tree=pages_tree,
            molecule_source=molecule_source,
        )
        try:
            result = requests.post(self.get_result_endpoint, json=asdict(data, dict_factory=int_enum_factory))
        except Exception:
            return {
                "status": StatusFlag.Error,
                "token": token,
                "message": "get result failed",
                "description": traceback.format_exc(),
            }
        try:
            return result.json()
        except json.decoder.JSONDecodeError:
            return {"status": StatusFlag.Error, "token": token, "message": result.text}

    def get_formatted(
        self,
        token: str,
        content: bool = False,
        objects: bool = False,
        pages_dict: bool = False,
        pages_tree: bool = False,
        molecule_source: bool = False,
        textual: FormatFlag = FormatFlag.Markdown,
        table: FormatFlag = FormatFlag.Markdown,
        molecule: FormatFlag = FormatFlag.Markdown,
        chart: FormatFlag = FormatFlag.Markdown,
        figure: FormatFlag = FormatFlag.Markdown,
        expression: FormatFlag = FormatFlag.Markdown,
        equation: FormatFlag = FormatFlag.Markdown,
    ):
        data = GetFormattedData(
            token=token,
            content=content,
            objects=objects,
            pages_dict=pages_dict,
            pages_tree=pages_tree,
            molecule_source=molecule_source,
            textual=textual,
            table=table,
            molecule=molecule,
            chart=chart,
            figure=figure,
            expression=expression,
            equation=equation,
        )
        try:
            result = requests.post(self.get_formatted_endpoint, json=asdict(data, dict_factory=int_enum_factory))
        except Exception:
            return {
                "status": StatusFlag.Error,
                "token": token,
                "message": "get formatted failed",
                "description": traceback.format_exc(),
            }
        try:
            return result.json()
        except json.decoder.JSONDecodeError:
            return {"status": StatusFlag.Error, "token": token, "message": result.text}
