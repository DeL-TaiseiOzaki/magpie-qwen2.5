from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import os
import gc
import torch  
import sys  
import random

# モデルとトークナイザーの初期化
tokenizer = AutoTokenizer.from_pretrained("AXCXEPT/EZO-Qwen2.5-32B-Instruct")
llm = LLM(model="AXCXEPT/EZO-Qwen2.5-32B-Instruct", gpu_memory_utilization=0.99, max_model_len=16384)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=640,
    top_p=0.8,
    top_k=40,
)

BATCH_SIZE = 10  # バッチサイズを適切に設定
SAVE_INTERVAL = 30  # 保存間隔

# システムプロンプトのリスト
SYSTEM_PROMPTS = [
    "あなたは好奇心旺盛で知識欲の高いアシスタントです。どんな質問にも詳細に答え、新しい視点を提供することを心がけてください。",
    "あなたは論理的で分析力に優れたアシスタントです。問題を細分化し、段階的に解決策を提示することを得意としています。",
    "あなたは創造性豊かで斬新なアイデアを生み出すアシスタントです。常識にとらわれない発想で問題解決に取り組んでください。",
    "あなたは温かみのある共感力の高いアシスタントです。相手の気持ちを理解し、寄り添うような問題解決と返答を心がけてください。",
    "あなたは効率と生産性を重視するアシスタントです。無駄を省いた簡潔で実用的な解決策を提供することを目指してください。",
    "あなたは歴史や文化に精通したアシスタントです。様々な時代や地域の知識を活かし、多角的な視点から回答してください。",
    "あなたは冷静沈着で公平なアシスタントです。感情に左右されず、客観的な事実に基づいた回答を提供してください。",
    "あなたは楽観的でユーモアのセンスがあるアシスタントです。難しい問題でも前向きに捉え、時には冗談を交えて問題回答してください。",
    "あなたは細部にこだわる完璧主義者のアシスタントです。丁寧で正確な情報提供を心がけ、些細な点も見逃さないようにしてください。",
    "あなたは柔軟性が高く適応力のあるアシスタントです。状況に応じて異なるアプローチを取り、多様な解決策を提示してください。"
]

def load_progress(filename="progress.json"):
    """進捗をロード"""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {"processed_count": 0}

def save_progress(progress, filename="progress.json"):
    """進捗を保存（トランザクション的）"""
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w") as f:
        json.dump(progress, f)
    os.rename(temp_filename, filename)

def append_outputs(data, filename="generated_sets.jsonl"):
    """生成されたデータを保存（アペンドモード）"""
    with open(filename, "a", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def clean_instruction(instruction):
    """指示文の冒頭をクリーンアップする関数"""
    # 指示文の冒頭30文字以内に \n\n があれば、その後の部分を使用
    split_index = instruction[:30].find("\n\n")
    if split_index != -1:
        return instruction[split_index + 2:].strip()  # \n\n の後ろを使用
    return instruction.strip()  # そのまま使用

def generate_texts(system_prompt, prompts):
    """バッチ処理でテキストを生成（同期）"""
    messages_batch = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = llm.generate(texts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def generate_pipeline(batch_size):
    """指示文→推論→回答→回答改良のパイプラインをバッチ処理で実行"""

    # 1. システムプロンプトをランダムに選択
    system_prompt = random.choice(SYSTEM_PROMPTS)

    # 2. 指示文の生成
    instruction_prompts = [
        "何らかのタスクを説明する簡潔な指示文を日本語で自由に作成してください。\n指示の内容以外は一切書かないでください。"
        for _ in range(batch_size)
    ]
    raw_instructions = generate_texts(system_prompt, instruction_prompts)

    # 3. 各指示文をクリーンアップ
    instructions = [clean_instruction(instruction) for instruction in raw_instructions]

    # 4. 推論手順の生成
    reasoning_prompts = [
        f"指示: {instruction}\n\nこの指示を達成するための論理的な推論手順を段階的に簡潔に説明してください。\n説明の内容以外は一切書かないでください。"
        for instruction in instructions
    ]
    reasonings = generate_texts(system_prompt, reasoning_prompts)

    # 5. 最初の回答の生成
    answer_prompts = [
        f"指示: {instruction}\n\n推論手順: {reasoning}\n\n推論手順に基づき、指示に対して簡潔に回答してください。\n回答の内容以外は一切書かないでください。"
        for instruction, reasoning in zip(instructions, reasonings)
    ]
    answers = generate_texts(system_prompt, answer_prompts)

    # 6. 回答の改良 (Self-Refine 機構)
    refine_prompts = [
        f"指示: {instruction}\n\n元の回答: {answer}\n\nこの回答の品質を高めるために、誤解を避けつつ、さらに簡潔で洗練された回答に改良してください。\n改良後の回答の内容以外は一切書かないでください。"
        for instruction, answer in zip(instructions, answers)
    ]
    refined_answers = generate_texts(system_prompt, refine_prompts)

    # 7. 生成結果をまとめる
    generated_sets = [
        {
            "instruction": instruction,
            "reasoning": reasoning,
            "initial_answer": answer,
            "refined_answer": refined_answer
        }
        for instruction, reasoning, answer, refined_answer in zip(instructions, reasonings, answers, refined_answers)
    ]

    return generated_sets

def clear_memory():
    """メモリをクリアする関数"""
    gc.collect()
    torch.cuda.empty_cache()
    # 強制的に未使用オブジェクトを削除
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                del obj
        except Exception:
            pass

def main():
    # 進捗の読み込み
    progress = load_progress()
    processed_count = progress["processed_count"]

    total_generations = 10000

    # バッファの初期化
    buffer = []

    try:
        while processed_count < total_generations:
            remaining = total_generations - processed_count
            batch_size = min(BATCH_SIZE, remaining)

            # 1バッチのデータを生成
            generated_sets = generate_pipeline(batch_size=batch_size)

            # バッファに追加
            buffer.extend(generated_sets)
            processed_count += len(generated_sets)

            # 進捗表示
            print(f"処理済み: {processed_count}/{total_generations}", end='\r')

            # バッファが十分に溜まったら保存
            if len(buffer) >= SAVE_INTERVAL:
                append_outputs(buffer)
                buffer = []  # バッファをクリア
                print(f"\n{processed_count}件処理しました。")
                save_progress({"processed_count": processed_count})

                # メモリをクリア
                clear_memory()

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        # 保存処理を行う（必要に応じて）
        if buffer:
            append_outputs(buffer)
            buffer = []
        save_progress({"processed_count": processed_count})
        # メモリをクリア
        clear_memory()
        raise  # 再度例外を投げて終了

    finally:
        # 最終的なバッファの保存
        if buffer:
            append_outputs(buffer)
            buffer = []

        print(f"\nすべての{total_generations}件の生成が完了しました。")
        # 最終的な進捗保存
        save_progress({"processed_count": processed_count})

        # メモリをクリア
        clear_memory()

if __name__ == "__main__":
    main()