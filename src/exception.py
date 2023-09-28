import sys
from src.logger import logging



def error_message_detail(error , error_detail : sys):
    _,_, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message  = f'Error Occured in Python Script Name {file_name} in Line no. {line_number} -> Error Detail : {str(error)} .'
    return error_message


class CustomException(Exception):
    def __init__(self , error_message , error_detail : sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error = error_message , error_detail= error_detail)

    def __str__(self): # this defaults prints whatever it returns , on the comman line interface when the custom exception occurs.
        return self.error_message 
    

