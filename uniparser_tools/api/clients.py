import json
import re
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
    ThirdPartyFormatter,
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
    pages: List[int] = None
    admin_debug: bool = False
    timeout: int = 1800
    table_cls: bool = False
    ordering_method: OrderingMethod = OrderingMethod.XYCutExp
    padding_snip: bool = True
    inplace_update: bool = False
    preset_layout: str = ""
    callback_url: str = None
    callback_secret: str = None


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
    pages: List[int] = None
    admin_debug: bool = False
    timeout: int = 1800
    table_cls: bool = False
    ordering_method: OrderingMethod = OrderingMethod.XYCutExp
    proxy: str = None
    inplace_update: bool = False
    preset_layout: str = ""
    callback_url: str = None
    callback_secret: str = None


@dataclass
class GetResultData:
    token: str
    return_half: bool
    content: bool
    objects: bool
    pages_dict: bool
    pages_tree: bool
    molecule_source: bool


@dataclass
class GetFormattedData:
    token: str
    return_half: bool
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
    marginalia: bool


@dataclass
class GetThirdPartyData:
    token: str
    formatter: ThirdPartyFormatter


class UniParserClient:
    def __init__(self, host: str, api_key: str):
        assert api_key, "api_key can not be empty"
        assert host.startswith("http"), "host must start with http or https"
        self.api_key = api_key
        self.user = uuid.uuid5(uuid.NAMESPACE_DNS, self.api_key)
        self.host = host.rstrip("/")

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

    @property
    def get_third_party_output_endpoint(self):
        return f"{self.host}/get-third-party-output"

    def to_token(self, task_id: str):
        token = uuid.uuid5(self.user, task_id).hex
        return token

    def validate_token(self, token: str):
        assert re.match(r"^[-\._?=&a-zA-Z0-9]{1,128}$", token), f"token: {token} contains illegal characters"

    def health(self):
        try:
            headers = {"X-API-Key": self.api_key}
            response = requests.get(f"{self.host}/health", headers=headers, timeout=30)
        except Exception:
            return {
                "status": StatusFlag.Error,
                "description": traceback.format_exc(),
            }
        if response.status_code >= 400:
            return {
                "status": "error",
                "http_status": response.status_code,
                "description": response.reason,
                "body": response.text,
            }
        try:
            return response.json()
        except json.decoder.JSONDecodeError:
            return {"status": StatusFlag.Error, "message": response.text}

    def version(self):
        try:
            headers = {"X-API-Key": self.api_key}
            response = requests.get(f"{self.host}/version", headers=headers, timeout=30)
        except Exception:
            return {
                "status": StatusFlag.Error,
                "description": traceback.format_exc(),
            }
        if response.status_code >= 400:
            return {
                "status": "error",
                "http_status": response.status_code,
                "description": response.reason,
                "body": response.text,
            }
        try:
            return response.json()
        except json.decoder.JSONDecodeError:
            return {"status": StatusFlag.Error, "message": response.text}

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
        ordering_method: OrderingMethod = OrderingMethod.GapTree,
        callback_url: str = None,
        callback_secret: str = None,
        admin_debug: bool = False,
        timeout: int = 1800,
        table_cls: bool = False,
        padding_snip: bool = True,
        inplace_update: bool = False,
        preset_layout: Union[str, list] = "",
        **kwargs,
    ):
        """
        sync: True=同步解析，该请求会在解析完成后才返回; False=异步解析，该请求会立即返回，解析结果需要通过GetResult接口获取
        callback_url: 异步解析完成后的回调地址
        callback_secret: 回调验证密钥
        """
        if not token:
            token = self.to_token(file_path)
        self.validate_token(token)
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
            admin_debug=admin_debug,
            timeout=timeout,
            table_cls=table_cls,
            ordering_method=ordering_method,
            padding_snip=padding_snip,
            inplace_update=inplace_update,
            preset_layout=preset_layout if isinstance(preset_layout, str) else json.dumps(preset_layout),
            callback_url=callback_url,
            callback_secret=callback_secret,
        )

        try:
            headers = {"X-API-Key": self.api_key}
            data = asdict(trigger_data, dict_factory=int_enum_factory)
            with open(file_path, "rb") as file:
                response = requests.post(
                    self.trigger_file_endpoint,
                    files={"file": file},
                    data=data,
                    headers=headers,
                    timeout=timeout,
                )
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
        ordering_method: OrderingMethod = OrderingMethod.GapTree,
        callback_url: str = None,
        callback_secret: str = None,
        admin_debug: bool = False,
        timeout: int = 1800,
        table_cls: bool = False,
        padding_snip: bool = True,
        inplace_update: bool = False,
        preset_layout: Union[str, list] = "",
        **kwargs,
    ):
        if not token:
            token = self.to_token(snip_path)
        self.validate_token(token)
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
            admin_debug=admin_debug,
            timeout=timeout,
            table_cls=table_cls,
            ordering_method=ordering_method,
            padding_snip=padding_snip,
            inplace_update=inplace_update,
            preset_layout=preset_layout if isinstance(preset_layout, str) else json.dumps(preset_layout),
            callback_url=callback_url,
            callback_secret=callback_secret,
        )

        try:
            headers = {"X-API-Key": self.api_key}
            with Image.open(snip_path) as image:
                img = dump_image_base64_str(image.convert("RGB"))
            data = {"img": img, **asdict(trigger_data, dict_factory=int_enum_factory)}
            result = requests.post(self.trigger_snip_endpoint, data=data, headers=headers, timeout=timeout)
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
        ordering_method: OrderingMethod = OrderingMethod.GapTree,
        proxy: str = None,
        callback_url: str = None,
        callback_secret: str = None,
        admin_debug: bool = False,
        timeout: int = 1800,
        table_cls: bool = False,
        inplace_update: bool = False,
        preset_layout: Union[str, list] = "",
        **kwargs,
    ):
        if not token:
            token = self.to_token(pdf_url)
        self.validate_token(token)
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
            admin_debug=admin_debug,
            timeout=timeout,
            table_cls=table_cls,
            ordering_method=ordering_method,
            proxy=proxy,
            inplace_update=inplace_update,
            preset_layout=preset_layout if isinstance(preset_layout, str) else json.dumps(preset_layout),
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
        try:
            headers = {"X-API-Key": self.api_key}
            data = asdict(trigger_data, dict_factory=int_enum_factory)
            result = requests.post(self.trigger_url_endpoint, json=data, headers=headers, timeout=timeout)
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
        return_half: bool = False,
    ):
        data = GetResultData(
            token=token,
            return_half=return_half,
            content=content,
            objects=objects,
            pages_dict=pages_dict,
            pages_tree=pages_tree,
            molecule_source=molecule_source,
        )
        try:
            headers = {"X-API-Key": self.api_key}
            data = asdict(data, dict_factory=int_enum_factory)
            result = requests.post(self.get_result_endpoint, json=data, headers=headers, timeout=30)
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
        marginalia: bool = False,
        return_half: bool = False,
    ):
        data = GetFormattedData(
            token=token,
            return_half=return_half,
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
            marginalia=marginalia,
        )
        try:
            headers = {"X-API-Key": self.api_key}
            data = asdict(data, dict_factory=int_enum_factory)
            result = requests.post(self.get_formatted_endpoint, json=data, headers=headers, timeout=30)
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

    def get_third_party_output(
        self,
        token: str,
        formatter: ThirdPartyFormatter = ThirdPartyFormatter.MinerU,
    ):
        data = GetThirdPartyData(token=token, formatter=formatter)
        try:
            headers = {"X-API-Key": self.api_key}
            data = asdict(data, dict_factory=int_enum_factory)
            result = requests.post(self.get_third_party_output_endpoint, json=data, headers=headers, timeout=30)
        except Exception:
            return {
                "status": StatusFlag.Error,
                "token": token,
                "message": "get third party output failed",
                "description": traceback.format_exc(),
            }
        try:
            return result.json()
        except json.decoder.JSONDecodeError:
            return {"status": StatusFlag.Error, "token": token, "message": result.text}
