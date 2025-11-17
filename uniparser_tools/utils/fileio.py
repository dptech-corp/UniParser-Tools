import re
from io import StringIO
from typing import List, Tuple

import pandas as pd
from pandas.io.html import _importers, _parser_dispatch, _validate_flavor
from pandas.io.html import read_html as pandas_read_html
from PIL import Image

from uniparser_tools.utils.log import get_root_logger


def read_html(io: StringIO) -> List[pd.DataFrame]:
    try:
        # modified from pandas.io.html.read_html v1.5.3
        _importers()

        # init args
        flavor = None
        match = ".+"
        attrs = None
        encoding = None
        displayed_only = True
        extract_links = None
        storage_options = None

        # copy form pandas.io.html._parse
        flavor = _validate_flavor(flavor)
        compiled_match = re.compile(match)  # you can pass a compiled regex here

        retained = None
        for flav in flavor:
            args = [
                io,
                compiled_match,
                attrs,
                encoding,
                displayed_only,
                extract_links,
            ]
            if pd.__version__ >= "2.1.0":
                args.append(storage_options)
            parser = _parser_dispatch(flav)
            p = parser(*args)

            try:
                tables: List[Tuple[List, List, List]] = p.parse_tables()
            except ValueError as caught:
                # if `io` is an io-like object, check if it's seekable
                # and try to rewind it before trying the next parser
                if hasattr(io, "seekable") and io.seekable():
                    io.seek(0)
                elif hasattr(io, "seekable") and not io.seekable():
                    # if we couldn't rewind it, let the user know
                    raise ValueError(
                        f"The flavor {flav} failed to parse your input. Since you passed a non-rewindable file object, we can't rewind it to try another parser. Try read_html() with a different flavor."
                    ) from caught

                retained = caught
            else:
                break
        else:
            assert retained is not None  # for mypy
            raise retained

        ret = []
        for th, tb, tf in tables:
            max_col = max([len(i) for i in th + tb + tf])
            # padding
            for i in range(len(th)):
                th[i] = th[i] + ["" for _ in range(max_col - len(th[i]))]
            for i in range(len(tb)):
                tb[i] = tb[i] + ["" for _ in range(max_col - len(tb[i]))]
            for i in range(len(tf)):
                tf[i] = tf[i] + ["" for _ in range(max_col - len(tf[i]))]
            # header
            if len(th) == 0:
                head = tb.pop(0)
            elif len(th) == 1:
                head = th[0]
            else:
                head = pd.MultiIndex.from_arrays(th)
            data = tb + tf
            ret.append(pd.DataFrame(data=data, columns=head))
        return ret
    except Exception:
        get_root_logger().exception("read_html error")
        return pandas_read_html(io)


def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图像的完整性
            return True
    except (IOError, Image.UnidentifiedImageError):
        return False
