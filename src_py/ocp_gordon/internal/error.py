"""
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2018 German Aerospace Center (DLR)

Created: 2018-08-06 Martin Siggel <Martin.Siggel@dlr.de>
"""

class ErrorCode:
    GENERIC_ERROR = 0
    MATH_ERROR = 1
    NULL_POINTER = 2
    INDEX_ERROR = 3
    INVALID_ARGUMENT = 4

class error(Exception):
    """
    Custom exception class mirroring the C++ error class.
    """
    def __init__(self, message: str = "", error_code: int = ErrorCode.GENERIC_ERROR):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self):
        return self.message

    def get_code(self) -> int:
        return self.error_code
