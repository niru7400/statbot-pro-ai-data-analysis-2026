import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from agent import create_agent, is_safe_question

CSV_PATH = "data/sample.csv"


def is_chart_question(question: str) -> bool:
    q = question.lower()
    keywords = ["plot", "chart", "graph", "pie", "bar", "line", "distribution"]
    return any(word in q for word in keywords)


def generate_chart(df, question: str):
    q = question.lower()
    os.makedirs("charts", exist_ok=True)

    filename = f"charts/chart_{int(time.time())}.png"

    plt.figure()

    # department distribution
    if "department" in q:
        if "pie" in q:
            df["department"].value_counts().plot(kind="pie", autopct="%1.1f%%")
            plt.ylabel("")
        else:
            df["department"].value_counts().plot(kind="bar")
            plt.xlabel("Department")
            plt.ylabel("Count")
        plt.title("Department Distribution")

    # semester distribution
    elif "semester" in q:
        if "pie" in q:
            df["semester"].value_counts().sort_index().plot(kind="pie", autopct="%1.1f%%")
            plt.ylabel("")
        else:
            df["semester"].value_counts().sort_index().plot(kind="bar")
            plt.xlabel("Semester")
            plt.ylabel("Count")
        plt.title("Semester Distribution")

    # math marks
    elif "math" in q:
        if "bar" in q:
            df["math_marks"].plot(kind="bar")
        else:
            df["math_marks"].plot(kind="line")
        plt.title("Math Marks")

    # science marks
    elif "science" in q:
        if "bar" in q:
            df["science_marks"].plot(kind="bar")
        else:
            df["science_marks"].plot(kind="line")
        plt.title("Science Marks")

    # english marks
    elif "english" in q:
        if "bar" in q:
            df["english_marks"].plot(kind="bar")
        else:
            df["english_marks"].plot(kind="line")
        plt.title("English Marks")

    # attendance
    elif "attendance" in q:
        if "bar" in q:
            df["attendance"].plot(kind="bar")
        else:
            df["attendance"].plot(kind="line")
        plt.title("Attendance")

    # fees paid
    elif "fees" in q:
        if "pie" in q:
            df["fees_paid"].value_counts().plot(kind="pie", autopct="%1.1f%%")
            plt.ylabel("")
        else:
            df["fees_paid"].value_counts().plot(kind="bar")
            plt.xlabel("Fees Paid")
            plt.ylabel("Count")
        plt.title("Fees Paid Distribution")

    else:
        plt.close()
        return "Chart request not understood. Example: Plot a pie chart of department distribution"

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return filename


def main():
    print("=== StatBot Pro ===")
    print("Type 'exit' to quit.")

    df = pd.read_csv(CSV_PATH)
    agent = create_agent(CSV_PATH)

    while True:
        question = input("\nAsk a question about the CSV: ").strip()

        if question.lower() == "exit":
            print("Exiting StatBot Pro...")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        if not is_safe_question(question):
            print("Blocked: Unsafe request detected.")
            continue

        try:
            if is_chart_question(question):
                result = generate_chart(df, question)
                print("\nAnswer:")
                print(result)
            else:
                result = agent.invoke({"input": question})
                print("\nAnswer:")
                print(result["output"])

        except Exception as e:
            print("\nError:", e)


if __name__ == "__main__":
    main()