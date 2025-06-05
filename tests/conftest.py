"""Pytest configuration and fixtures for Spark testing."""
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """
    Creates a Spark session for testing.
    
    This fixture is session-scoped, meaning it will be created once per test session
    and shared across all tests. The session will be automatically stopped after
    all tests complete.
    """
    spark = (SparkSession.builder
            .master("local[2]")  # Use 2 local cores
            .appName("pyspark-transform-registry-test")
            .config("spark.sql.shuffle.partitions", "2")  # Minimize partitions for testing
            .config("spark.default.parallelism", "2")
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")  # Allow security manager
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")  # Allow security manager
            .config("spark.authenticate", "false")  # Disable authentication
            .config("spark.ui.enabled", "false")  # Disable UI for testing
            .config("spark.driver.bindAddress", "127.0.0.1")  # Explicitly bind to localhost
            .getOrCreate())
    
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