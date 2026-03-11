"""Example: Evaluate model quality for schema extraction.

This script demonstrates how to use the evaluate_models() function
to compare different models (Gemma, GPT OSS, Llama, etc.) on their
ability to extract HTML schemas.

Usage:
    python examples/evaluate_models_example.py
"""

from earthquakes_parser.parser.html_schema_judge import HTMLSchemaJudge


def main():
    """Run model evaluation example."""
    print("=" * 70)
    print("Model Evaluation Example")
    print("=" * 70)
    print("\nThis will:")
    print("1. Load all schema-*.json files from data/")
    print("2. Test each model's selectors on real HTML")
    print("3. Calculate metrics: Success Rate, Precision, GPT Quality")
    print("4. Compute final scores out of 100")
    print("5. Save results to data/model_evaluation.json")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to start...\n")

    import time

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    # Initialize judge with your LLM server
    judge = HTMLSchemaJudge(
        openai_base_url="http://192.168.8.22:9999/v1",
        openai_api_key="api-key",  # pragma: allowlist secret
        model="gpt-4",
        max_tokens=1000,
    )

    # Run evaluation
    results = judge.evaluate_models()

    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nEvaluated {len(results.get('models', {}))} models")
    print("\nResults saved to: data/model_evaluation.json")
    print("\nTop 3 models:")

    sorted_models = sorted(
        results.get("models", {}).items(),
        key=lambda x: x[1]["final_score"],
        reverse=True,
    )

    for i, (model_name, metrics) in enumerate(sorted_models[:3], 1):
        print(f"{i}. {model_name}: {metrics['final_score']:.2f}/100")


if __name__ == "__main__":
    main()
