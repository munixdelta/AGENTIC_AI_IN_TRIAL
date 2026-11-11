
from concurrent.futures import ThreadPoolExecutor
import time


def calculate_percentage(number, percentage):

    print(f"Starting: {percentage}% of {number}")

    time.sleep(2)

    result = number * percentage / 100

    print(f"Completed: {result}")

    return result


numbers = [
    (200, 10),
    (500, 20),
    (100, 30)
]


start_time = time.time()


with ThreadPoolExecutor(max_workers=3) as executor:

    futures = [
        executor.submit(
            calculate_percentage,
            number,
            percentage
        )
        for number, percentage in numbers
    ]

    results = [future.result() for future in futures]


end_time = time.time()


print("\nResults:", results)

print(
    "Total execution time:",
    round(end_time - start_time, 2),
    "seconds"
)
