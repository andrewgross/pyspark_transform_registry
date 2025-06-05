"""Pytest configuration and fixtures for Spark testing."""

import subprocess
import tempfile
import time

import mlflow
import pytest
import requests
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def mlflow_server():
    """
    Starts a local MLflow tracking server for testing.
    Uses a TemporaryDirectory context manager for automatic cleanup.
    """
    with tempfile.TemporaryDirectory() as mlflow_dir:
        port = 5000  # We'll try this port first
        while True:
            try:
                process = subprocess.Popen(
                    [
                        "mlflow",
                        "server",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--backend-store-uri",
                        f"sqlite:///{mlflow_dir}/mlflow.db",
                        "--default-artifact-root",
                        mlflow_dir,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Wait for server to start
                for _ in range(30):
                    try:
                        response = requests.get(f"http://127.0.0.1:{port}/health")
                        if response.status_code == 200:
                            break
                    except requests.exceptions.ConnectionError:
                        time.sleep(1)
                else:
                    process.terminate()
                    raise Exception("MLflow server failed to start")
                mlflow.set_tracking_uri(f"http://127.0.0.1:{port}")
                yield f"http://127.0.0.1:{port}"
                process.terminate()
                process.wait()
                break
            except Exception as e:
                if port >= 5010:
                    raise Exception(
                        f"Failed to start MLflow server after trying multiple ports: {e}",
                    )
                port += 1
                continue


@pytest.fixture(scope="session")
def spark_session():
    """
    Creates a Spark session for testing.

    This fixture is session-scoped, meaning it will be created once per test session
    and shared across all tests. The session will be automatically stopped after
    all tests complete.
    """
    spark = (
        SparkSession.builder.master("local[2]")  # Use 2 local cores
        .appName("pyspark-transform-registry-test")
        .config("spark.sql.shuffle.partitions", "2")  # Minimize partitions for testing
        .config("spark.default.parallelism", "2")
        .config(
            "spark.driver.extraJavaOptions",
            "-Djava.security.manager=allow",
        )  # Allow security manager
        .config(
            "spark.executor.extraJavaOptions",
            "-Djava.security.manager=allow",
        )  # Allow security manager
        .config("spark.authenticate", "false")  # Disable authentication
        .config("spark.ui.enabled", "false")  # Disable UI for testing
        .config("spark.driver.bindAddress", "127.0.0.1")  # Explicitly bind to localhost
        .getOrCreate()
    )

    # Set log level to WARN to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    yield spark

    # Cleanup
    spark.stop()


@pytest.fixture(scope="function")
def spark(spark_session):
    """
    Provides a clean Spark session for each test function.

    This fixture is function-scoped and uses the session-scoped spark_session
    to create a new session for each test. This ensures test isolation while
    maintaining efficiency.
    """
    # Clear any existing tables/views
    for table in spark_session.catalog.listTables():
        spark_session.catalog.dropTempView(table.name)

    yield spark_session
