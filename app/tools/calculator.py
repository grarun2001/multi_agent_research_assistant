def calculate(expression: str) -> str:
    """
    Safely evaluate a basic mathematical expression.
    """

    allowed = set("0123456789+-*/(). ")

    if not expression:
        return "Empty expression."

    if not all(char in allowed for char in expression):
        return "Invalid mathematical expression."

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except ZeroDivisionError:
        return "Cannot divide by zero."

    except Exception as e:
        return f"Calculation error: {str(e)}"