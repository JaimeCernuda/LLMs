import multiprocessing
import os
import system_metrics  # Import the module

def worker_function(barrier, result_dict, key, func, core_id, stop_event, *args):
    """Worker function that executes after barrier synchronization until stop_event is set."""
    set_cpu_affinity(core_id)
    set_high_priority()
    
    while not stop_event.is_set():  # Check if stop signal is set
        barrier.wait()  # Wait for the main process to trigger execution
        result_dict[key] = func(*args)  # Store result in shared dictionary
        barrier.wait()  # Signal completion

def set_cpu_affinity(core_id):
    """Pin the process to a specific CPU core."""
    try:
        os.sched_setaffinity(0, {core_id})
        print(f"Process pinned to CPU {core_id}")
    except AttributeError:
        print("CPU affinity not supported.")

def set_high_priority():
    """Set the process to higher priority (nice -10)."""
    try:
        os.nice(-10)
        print("Increased process priority.")
    except PermissionError:
        print("Need sudo to change priority.")

class SystemMonitor:
    def __init__(self, tasks):
        self.manager = multiprocessing.Manager()
        self.results = self.manager.dict()
        self.tasks = tasks
        self.barrier = multiprocessing.Barrier(len(tasks) + 1)  # Main process included
        self.processes = []
        self.cpu_cores = os.cpu_count()
        self.stop_event = multiprocessing.Event()  # Add stop event

    def start_monitoring(self):
        """Initialize and start monitoring processes."""
        for i, (key, func) in enumerate(self.tasks.items()):
            core_id = self.cpu_cores - 1 - i % self.cpu_cores  # Assign from highest to lowest core
            process = multiprocessing.Process(
                target=worker_function, args=(self.barrier, self.results, key, func, core_id, self.stop_event)
            )
            process.start()
            self.processes.append(process)

    def collect_metrics(self):
        """Trigger execution and collect results."""
        self.barrier.wait()  # Signal all workers to execute
        self.barrier.wait()  # Wait for them to complete
        return dict(self.results)

    def stop_monitoring(self):
        """Gracefully terminate all monitoring processes."""
        self.stop_event.set()  # Signal all workers to exit cleanly
        self.collect_metrics()
        for process in self.processes:
            process.join()  # Ensure each process finishes execution before exit
        self.manager.shutdown()