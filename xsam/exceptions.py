class StructuredException(Exception):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details

    def __str__(self):
        message = super().__str__()
        if self.details is not None:
            details_string = [f"{key}={value}" for key, value in self.details.items()]
            message = f"{message} {details_string}."
        return message


class XsamException(StructuredException):
    """Custom exception class associated with xsam pacakge."""

    pass
