import luigi
import os
import tarfile
import gzip
import shutil
import requests
import pandas as pd
import io
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class DownloadDataset(luigi.Task):
    dataset_name = luigi.Parameter()
    output_dir = luigi.Parameter(default="data")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_dir, f"{self.dataset_name}_RAW.tar"))

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # Прямая ссылка для скачивания
        url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={self.dataset_name}&format=file"
        print(f"Загрузка файла по ссылке{url}")

        # Скачивание файла
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Не удалось загрузить: {response.status_code}")

        # Сохранение файла
        with open(self.output().path, "wb") as f:
            f.write(response.content)
        print(f"Файл загружен и сохранен в {self.output().path}")


class ExtractAndProcessFiles(luigi.Task):
    dataset_name = luigi.Parameter()
    output_dir = luigi.Parameter(default="data")

    def requires(self):
        return DownloadDataset(dataset_name=self.dataset_name, output_dir=self.output_dir)

    def output(self):
        # Указываем директорию, в которой будут обработанные файлы
        return luigi.LocalTarget(os.path.join(self.output_dir, self.dataset_name, "processed_done.txt"))

    def run(self):
        main_dir = Path(self.output_dir) / self.dataset_name
        os.makedirs(main_dir, exist_ok=True)

        # Извлечение основного архива
        tar_path = self.input().path
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(main_dir)
        print(f"Распаковка архива {main_dir}")

        # Обработка каждого .gz файла
        for gz_file in main_dir.glob("*.gz"):
            extracted_file_path = gz_file.with_suffix("")  # Файл без .gz расширения

            # Создание папки для извлеченного файла
            extracted_dir = main_dir / extracted_file_path.stem
            os.makedirs(extracted_dir, exist_ok=True)

            print(f"Processing: {gz_file}")
            # Распаковка .gz файла
            with gzip.open(gz_file, "rb") as f_in, open(extracted_dir / extracted_file_path.name, "wb") as f_out:
                f_out.write(f_in.read())

            print(f"Extracted {gz_file} to {extracted_dir}")
            # Удаляем исходный .gz файл
            gz_file.unlink()

            # Разделяем текстовый файл на таблицы
            text_file = extracted_dir / extracted_file_path.name
            self.process_text_file(text_file)

        # Создаем файл-сигнал завершения
        with self.output().open("w") as f:
            f.write("Processing completed.")

    def process_text_file(self, file_path):
        """
        Обработка текстового файла: разделение на таблицы и сохранение в отдельные TSV-файлы.
        """
        dfs = {}
        with open(file_path, "r", encoding="utf-8") as f:
            write_key = None
            fio = io.StringIO()
            for line in f:
                if line.startswith("["):
                    if write_key:
                        fio.seek(0)
                        header = None if write_key == "Heading" else "infer"
                        dfs[write_key] = pd.read_csv(fio, sep="\t", header=header)

                    # Подготавливаем новый ключ и StringIO
                    fio = io.StringIO()
                    write_key = line.strip("[]\n")
                    continue
                if write_key:
                    fio.write(line)

            # Обрабатываем последнюю таблицу
            fio.seek(0)
            if write_key:
                dfs[write_key] = pd.read_csv(fio, sep="\t")

        # Сохраняем каждую таблицу в отдельный файл
        for key, df in dfs.items():
            output_file = file_path.parent / f"{key}.tsv"
            df.to_csv(output_file, sep="\t", index=False)
            print(f"Сохранение таблицы {key}  {output_file}")

class TrimProbesTable(luigi.Task):
    dataset_name = luigi.Parameter()
    output_dir = luigi.Parameter(default="data")

    def requires(self):
        return ExtractAndProcessFiles(dataset_name=self.dataset_name, output_dir=self.output_dir)

    def output(self):
        # Указываем список выходных файлов, один для каждого `Probes.tsv`
        main_dir = Path(self.output_dir) / self.dataset_name
        return [
            luigi.LocalTarget(probes_file.with_name(f"{probes_file.stem}_trimmed.tsv"))
            for probes_file in main_dir.rglob("Probes.tsv")
        ]

    def run(self):
        main_dir = Path(self.output_dir) / self.dataset_name

        # Поиск всех файлов `Probes.tsv`
        probes_files = list(main_dir.rglob("Probes.tsv"))
        if not probes_files:
            raise FileNotFoundError(f"Файлы Probes.tsv не найдены в подкаталогах {main_dir}")

        print(f"Найдено {len(probes_files)} файлов Probes.tsv: {probes_files}")

        # Удаляем указанные колонки и сохраняем обработанные файлы
        columns_to_remove = [
            "Definition",
            "Ontology_Component",
            "Ontology_Process",
            "Ontology_Function",
            "Synonyms",
            "Obsolete_Probe_Id",
            "Probe_Sequence",
        ]

        for probes_file in probes_files:
            print(f"Обрабатываю файл: {probes_file}")
            df = pd.read_csv(probes_file, sep="\t")
            trimmed_df = df.drop(columns=columns_to_remove, errors="ignore")

            # Генерируем имя выходного файла
            trimmed_file_path = probes_file.with_name(f"{probes_file.stem}_trimmed.tsv")
            trimmed_df.to_csv(trimmed_file_path, sep="\t", index=False)
            print(f"Сохранен урезанный файл: {trimmed_file_path}")

class CleanupOriginalFiles(luigi.Task):
    dataset_name = luigi.Parameter()
    output_dir = luigi.Parameter(default="data")

    def requires(self):
        return TrimProbesTable(dataset_name=self.dataset_name, output_dir=self.output_dir)

    def output(self):
        # Индикатор завершения задачи
        return luigi.LocalTarget(
            Path(self.output_dir) / self.dataset_name / "cleanup_done.txt"
        )

    def run(self):
        main_dir = Path(self.output_dir) / self.dataset_name
        deleted_files = []

        # Удаляем .txt файлы
        for txt_file in main_dir.rglob("*.txt"):
            print(f"Удаляю исходный текстовый файл: {txt_file}")
            txt_file.unlink()
            deleted_files.append(txt_file)

        # Удаляем .gz файлы
        for gz_file in main_dir.rglob("*.gz"):
            print(f"Удаляю оставшийся .gz файл: {gz_file}")
            gz_file.unlink()
            deleted_files.append(gz_file)

        # Удаляем пустые директории
        for dir_path in main_dir.glob("**/"):
            if not any(dir_path.iterdir()):  # Если директория пуста
                print(f"Удаляю пустую директорию: {dir_path}")
                dir_path.rmdir()

        # Логирование завершения
        print(f"Удалено файлов: {len(deleted_files)}")

        # Создаём файл-сигнал о завершении задачи
        with self.output().open("w") as f:
            f.write("Cleanup completed.\n")

if __name__ == "__main__":
    luigi.run()
