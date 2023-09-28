FROM amazon/aws-lambda-python:3.9

RUN /var/lang/bin/python3.9 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

COPY ./code ${LAMBDA_TASK_ROOT}

CMD ["handler.lambda_handler"]