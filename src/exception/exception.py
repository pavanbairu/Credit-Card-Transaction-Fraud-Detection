import os
import sys
from src.logger.logging import logging

def error_message_details(error, error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno

    return f"The error has been occured in script: {file_name} and line number: {line_no} and the error: {error}"


class CreditFraudException(Exception):

    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error = error_message_details(error_message, error_details)

    def __str__(self):
        return self.error
    