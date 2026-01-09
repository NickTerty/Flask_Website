from flask import Flask, render_template, request

app = Flask(__name__)

def transpose_matrix(A):
    return list(map(list, zip(*A)))

def format_number(x, decimals=2):
    """
    If the number is integer, show it as an integer
    """
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    else:
        return f"{x:.{decimals}f}".rstrip('0').rstrip('.')

def format_matrix(M, decimals=2):
    return [
        [format_number(value, decimals) for value in row]
        for row in M
    ]

def matrix_properties(A):
    rows = len(A)
    cols = len(A[0])
    return {
        "rows": rows,
        "cols": cols,
        "is_square": rows == cols
    }

def parse_matrix(text):
    matrix = []
    lines = text.strip().split('\n')

    for line in lines:
        row = list(map(float, line.strip().split()))
        matrix.append(row)

    # 檢查每一列長度是否一致
    row_length = len(matrix[0])
    for row in matrix:
        if len(row) != row_length:
            raise ValueError("Each row must have the same number of elements.")

    return matrix


def matrix_multiply(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])

    if n != n2:
        raise ValueError(
            "The number of columns of Matrix A must equal "
            "the number of rows of Matrix B."
        )

    result = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result

def determinant(A):
    n = len(A)

    # 1x1
    if n == 1:
        return A[0][0]

    # 2x2
    if n == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]

    det = 0
    for col in range(n):
        # 建立 minor matrix
        minor = [
            row[:col] + row[col+1:]
            for row in A[1:]
        ]
        det += ((-1) ** col) * A[0][col] * determinant(minor)

    return det


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/matrix', methods=['GET', 'POST'])
def matrix():
    result = None
    error = None
    matrixA_text = ""
    matrixB_text = ""
    propA = None
    propB = None

    if request.method == 'POST':
        matrixA_text = request.form['matrixA']
        operation = request.form['operation']

        try:
            A = parse_matrix(matrixA_text)
            propA = matrix_properties(A)

            if operation == 'transpose':
                raw_result = transpose_matrix(A)
                result = format_matrix(raw_result)

            elif operation == 'multiply':
                matrixB_text = request.form['matrixB']
                B = parse_matrix(matrixB_text)
                propB = matrix_properties(B)
                raw_result = matrix_multiply(A, B)
                result = format_matrix(raw_result)
            
            elif operation == 'determinant':
                if not propA["is_square"]:
                    raise ValueError("Determinant is only defined for square matrices.")
                det_value = determinant(A)
                result = format_number(det_value)


        except ValueError as e:
            error = f"Input error: {e}"
        except Exception:
            error = "Invalid input format. Please check your matrices."

    return render_template(
        'matrix.html',
        result=result,
        error=error,
        matrixA_text=matrixA_text,
        matrixB_text=matrixB_text,
        propA=propA,
        propB=propB
    )


if __name__ == '__main__':
    app.run(debug=True)
