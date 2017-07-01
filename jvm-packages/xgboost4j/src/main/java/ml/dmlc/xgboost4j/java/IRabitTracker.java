package ml.dmlc.xgboost4j.java;

import java.util.Map;

/**
 * Interface for Rabit tracker implementations with three public methods:
 *
 *  - start(): Start the Rabit tracker awaiting for worker connections.
 *  - getWorkerEnvs(): Return the environment variables needed to initialize Rabit clients.
 *  - waitFor(timeout): Wait for the task execution by the worker nodes for at most `timeout`
 *  milliseconds.
 *
 * Each implementation is expected to implement a callback function
 *
 *    public void uncaughtException(Threat t, Throwable e) { ... }
 *
 * to interrupt waitFor() in order to prevent the tracker from hanging indefinitely.
 *
 * The Rabit tracker handles connections from distributed workers, assigns ranks to workers, and
 * brokers connections between workers.
 */
public interface IRabitTracker extends Thread.UncaughtExceptionHandler {
  enum TrackerStatus {
    SUCCESS(0), INTERRUPTED(1), TIMEOUT(2), FAILURE(3);

    private int statusCode;

    TrackerStatus(int statusCode) {
      this.statusCode = statusCode;
    }

    public int getStatusCode() {
      return this.statusCode;
    }
  }

  Map<String, String> getWorkerEnvs();
  boolean start();
  void stop();
  // taskExecutionTimeout has no effect in current version of XGBoost.
  int waitFor(long taskExecutionTimeout);
}
