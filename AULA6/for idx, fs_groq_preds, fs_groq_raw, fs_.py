for idx, fs_groq_preds, fs_groq_raw, fs_groq_cost, fs_mistral_preds, fs_mistral_raw, fs_mistral_cost in zip(range(len(biased_set_fs_groq_preds)), biased_set_fs_groq_preds, biased_set_fs_groq_raw, biased_set_fs_groq_cost, biased_set_fs_mistral_preds, biased_set_fs_mistral_raw, biased_set_fs_mistral_cost):
    # 5) display evaluation tables
    # 6) print the raw outputs to inspect formatting
    rows_groq = evaluate_predictions(fs_groq_preds, gold_fs)
    rows_mistral = evaluate_predictions(fs_mistral_preds, gold_fs)

    fs_preds[f"{idx+1}_per_label"] = {
        "groq": fs_groq_preds,
        "mistral": fs_mistral_preds,
    }

    print(f"Raw outputs for few-shot with {idx+1} examples per label, and cost:")
    print("Groq raw outputs:{0}, Cost: ${1:.10f}".format(fs_groq_raw, fs_groq_cost))
    print("Mistral raw outputs:{0}, Cost: ${1:.10f}".format(fs_mistral_raw, fs_mistral_cost))