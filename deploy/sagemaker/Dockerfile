# Training image for `lineage cloud apply`. Build it from the repository root and push
# it to ECR, then pin it BY DIGEST in [cloud].training_image:
#
#   docker build -f deploy/sagemaker/Dockerfile \
#     --build-arg LINEAGE_CODE_COMMIT=$(git rev-parse HEAD) \
#     -t <acct>.dkr.ecr.<region>.amazonaws.com/lineage:train .
#   docker push ... && docker inspect --format '{{index .RepoDigests 0}}' ...
#
# The job runs with network isolation: everything it needs is in this image or in the
# input channels (dataset, base model).
FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 GIT_PYTHON_REFRESH=quiet
WORKDIR /opt/lineage
COPY requirements/ml.txt requirements/ml.txt
RUN pip install --require-hashes -r requirements/ml.txt
COPY pyproject.toml README.md LICENSE ./
COPY src src
# Same relative layout as the repository, so the lock file path in lineage.toml resolves
# and the run records the lock hash it was installed from.
COPY examples/triage/lineage.toml examples/triage/lineage.toml
RUN pip install --no-deps . && useradd --create-home lineage && chown -R lineage /opt/lineage
# No .git in the image: the commit it was built from is passed in and recorded by runs.
ARG LINEAGE_CODE_COMMIT=unknown
ENV LINEAGE_CODE_COMMIT=${LINEAGE_CODE_COMMIT}
COPY deploy/sagemaker/entry.py /opt/lineage/entry.py
USER lineage
ENTRYPOINT ["python", "/opt/lineage/entry.py"]
