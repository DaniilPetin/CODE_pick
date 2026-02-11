!pip install segyio
import numpy as np
import matplotlib.pyplot as plt
import segyio
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

class SegyAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ SEG-Y файлов")

        # Создание Canvas и Scrollbar
        self.canvas = tk.Canvas(root)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        # Настройка прокрутки
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Переменные для хранения данных
        self.file_path = None
        self.traces = None
        self.samples = None
        self.trace_number = 0
        self.show_envelope = False
        self.hide_trace = False
        self.mode = "peaks"
        self.peak_times = None

        # Создание фреймов внутри прокручиваемой области
        self.mode_frame = tk.Frame(self.scrollable_frame)
        self.mode_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.input_frame = tk.Frame(self.scrollable_frame)
        self.input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.plot_frame = tk.Frame(self.scrollable_frame)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.freq_frame = tk.Frame(self.scrollable_frame)
        self.freq_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1, pady=10)

        self.amp_frame = tk.Frame(self.scrollable_frame)
        self.amp_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1, pady=10)

        self.result_frame = tk.Frame(self.scrollable_frame)
        self.result_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        # Кнопки выбора режима
        self.mode_peaks_button = tk.Button(self.mode_frame, text="Отметка пиков", command=lambda: self.set_mode("peaks"))
        self.mode_peaks_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.mode_onsets_button = tk.Button(self.mode_frame, text="Отметка времен вступления", command=lambda: self.set_mode("onsets"))
        self.mode_onsets_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Поля ввода для режима "Отметка пиков"
        self.peaks_threshold_label = tk.Label(self.input_frame, text="Порог амплитуды:")
        self.peaks_threshold_label.grid(row=0, column=0, padx=5, pady=5)
        self.peaks_threshold_entry = tk.Entry(self.input_frame)
        self.peaks_threshold_entry.grid(row=0, column=1, padx=5, pady=5)
        self.peaks_threshold_entry.insert(0, "1.5")

        self.peaks_distance_label = tk.Label(self.input_frame, text="Минимальное расстояние между пиками:")
        self.peaks_distance_label.grid(row=1, column=0, padx=5, pady=5)
        self.peaks_distance_entry = tk.Entry(self.input_frame)
        self.peaks_distance_entry.grid(row=1, column=1, padx=5, pady=5)
        self.peaks_distance_entry.insert(0, "40")

        # Поля ввода для режима "Отметка времен вступления"
        self.onsets_threshold_label = tk.Label(self.input_frame, text="Порог амплитуды:")
        self.onsets_threshold_label.grid(row=2, column=0, padx=5, pady=5)
        self.onsets_threshold_entry = tk.Entry(self.input_frame)
        self.onsets_threshold_entry.grid(row=2, column=1, padx=5, pady=5)
        self.onsets_threshold_entry.insert(0, "100")

        self.onsets_interval_label = tk.Label(self.input_frame, text="Минимальный интервал (мс):")
        self.onsets_interval_label.grid(row=3, column=0, padx=5, pady=5)
        self.onsets_interval_entry = tk.Entry(self.input_frame)
        self.onsets_interval_entry.grid(row=3, column=1, padx=5, pady=5)
        self.onsets_interval_entry.insert(0, "65")

        # Общие поля ввода
        self.trace_number_label = tk.Label(self.input_frame, text="Номер трассы:")
        self.trace_number_label.grid(row=4, column=0, padx=5, pady=5)
        self.trace_number_entry = tk.Entry(self.input_frame)
        self.trace_number_entry.grid(row=4, column=1, padx=5, pady=5)
        self.trace_number_entry.insert(0, "0")

        self.min_time_label = tk.Label(self.input_frame, text="Минимальное время (мс):")
        self.min_time_label.grid(row=5, column=0, padx=5, pady=5)
        self.min_time_entry = tk.Entry(self.input_frame)
        self.min_time_entry.grid(row=5, column=1, padx=5, pady=5)
        self.min_time_entry.insert(0, "0")

        self.max_time_label = tk.Label(self.input_frame, text="Максимальное время (мс):")
        self.max_time_label.grid(row=6, column=0, padx=5, pady=5)
        self.max_time_entry = tk.Entry(self.input_frame)
        self.max_time_entry.grid(row=6, column=1, padx=5, pady=5)
        self.max_time_entry.insert(0, "1000")

        self.open_button = tk.Button(self.input_frame, text="Открыть файл", command=self.open_file)
        self.open_button.grid(row=7, column=0, columnspan=2, pady=10)

        self.recalculate_button = tk.Button(self.input_frame, text="Пересчитать", command=self.recalculate, state=tk.DISABLED)
        self.recalculate_button.grid(row=8, column=0, columnspan=2, pady=10)

        self.envelope_button = tk.Button(self.input_frame, text="Добавить огибающую", command=self.toggle_envelope, state=tk.DISABLED)
        self.envelope_button.grid(row=9, column=0, columnspan=2, pady=10)

        self.hide_trace_button = tk.Button(self.input_frame, text="Спрятать трассу", command=self.toggle_trace, state=tk.DISABLED)
        self.hide_trace_button.grid(row=10, column=0, columnspan=2, pady=10)

        self.save_button = tk.Button(self.input_frame, text="Сохранить результаты", state=tk.DISABLED)
        self.save_button.grid(row=11, column=0, columnspan=2, pady=10)

        # Новая кнопка для экспорта бинарных данных
        self.export_binary_button = tk.Button(self.input_frame, text="Выгрузить 0 и 1", command=self.export_binary_data, state=tk.DISABLED)
        self.export_binary_button.grid(row=12, column=0, columnspan=2, pady=10)

        self.result_text = scrolledtext.ScrolledText(self.result_frame, height=10, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=1)

        # Инициализация режима
        self.set_mode("peaks")

    def set_mode(self, mode):
        self.mode = mode
        if mode == "peaks":
            self.peaks_threshold_label.grid()
            self.peaks_threshold_entry.grid()
            self.peaks_distance_label.grid()
            self.peaks_distance_entry.grid()
            self.onsets_threshold_label.grid_remove()
            self.onsets_threshold_entry.grid_remove()
            self.onsets_interval_label.grid_remove()
            self.onsets_interval_entry.grid_remove()
        elif mode == "onsets":
            self.peaks_threshold_label.grid_remove()
            self.peaks_threshold_entry.grid_remove()
            self.peaks_distance_label.grid_remove()
            self.peaks_distance_entry.grid_remove()
            self.onsets_threshold_label.grid()
            self.onsets_threshold_entry.grid()
            self.onsets_interval_label.grid()
            self.onsets_interval_entry.grid()
        self.recalculate()

    def open_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("SEGY files", "*.sgy")])
        if self.file_path:
            try:
                with segyio.open(self.file_path, "r") as f:
                    self.traces = segyio.tools.collect(f.trace[:])
                    self.samples = f.samples
                    self.recalculate_button.config(state=tk.NORMAL)
                    self.save_button.config(state=tk.NORMAL)
                    self.envelope_button.config(state=tk.NORMAL)
                    self.hide_trace_button.config(state=tk.NORMAL)
                    self.export_binary_button.config(state=tk.NORMAL)
                    self.recalculate()
            except Exception as e:
                messagebox.showerror("Ошибка", f'Ошибка при чтении файла: {e}')

    def recalculate(self):
        if self.traces is None:
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        for widget in self.freq_frame.winfo_children():
            widget.destroy()
        for widget in self.amp_frame.winfo_children():
            widget.destroy()

        trace_number = int(self.trace_number_entry.get())
        min_time = float(self.min_time_entry.get())
        max_time = float(self.max_time_entry.get())

        if trace_number >= self.traces.shape[0] or trace_number < 0:
            messagebox.showerror("Ошибка", "Некорректный номер трассы")
            return

        trace = self.traces[trace_number, :]

        if self.mode == "peaks":
            amplitude_threshold = float(self.peaks_threshold_entry.get())
            min_distance = int(self.peaks_distance_entry.get())
            peaks, _ = find_peaks(trace, height=amplitude_threshold, distance=min_distance)
            peaks_filtered = np.copy(peaks)
            for i, idx in enumerate(peaks):
                if ((self.samples[idx] < min_time) | (self.samples[idx] > max_time)):
                    peaks_filtered[i] = 0
                else:
                    peaks_filtered[i] = idx
            peaks_filtered = peaks_filtered[peaks_filtered > 0]
            self.peak_times = self.samples[peaks_filtered]
        elif self.mode == "onsets":
            amplitude_threshold = float(self.onsets_threshold_entry.get())
            min_interval = int(self.onsets_interval_entry.get())
            self.peak_times = self.find_impulse_times(trace, amplitude_threshold, min_interval)
            peaks_filtered = [int(t / (self.samples[1] - self.samples[0])) for t in self.peak_times]

        # Рассчет интервалов, частот и амплитуд
        intervals = np.diff(self.peak_times) if len(self.peak_times) > 1 else np.array([])
        freqs = 1 / (intervals / 1000) if len(intervals) > 0 else np.array([])
        amplitudes = trace[peaks_filtered] if len(peaks_filtered) > 0 else np.array([])

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, 'Максимальная амплитуда: ' + str(round(trace.max(), 5)) + '\n')
        self.result_text.insert(tk.END, 'Минимальная амплитуда: ' + str(round(trace.min(), 5)) + '\n')
        self.result_text.insert(tk.END, 'Времена пиков, мс: ' + str(self.peak_times) + '\n')
        if len(intervals) > 0:
            self.result_text.insert(tk.END, 'Интервалы времени между пиками, мс: ' + str(intervals) + '\n')
            self.result_text.insert(tk.END, 'Частоты, Гц: ' + str(freqs) + '\n')
        if len(amplitudes) > 0:
            self.result_text.insert(tk.END, 'Амплитуды пиков: ' + str(amplitudes) + '\n')
        self.result_text.config(state=tk.DISABLED)

        # Визуализация трассы с пиками
        fig_trace = self.visualize_trace_with_peaks(trace, peaks_filtered, min_time, max_time)
        canvas_trace = FigureCanvasTkAgg(fig_trace, master=self.plot_frame)
        canvas_trace.draw()
        canvas_trace.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Визуализация графика частот
        if len(self.peak_times) > 1:
            fig_freq = self.plot_frequencies(self.peak_times[1:], freqs, min_time, max_time)
            canvas_freq = FigureCanvasTkAgg(fig_freq, master=self.freq_frame)
            canvas_freq.draw()
            canvas_freq.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Визуализация графика амплитуд
        if len(self.peak_times) > 0:
            fig_amp = self.plot_amplitudes(self.peak_times, amplitudes, min_time, max_time)
            canvas_amp = FigureCanvasTkAgg(fig_amp, master=self.amp_frame)
            canvas_amp.draw()
            canvas_amp.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.save_button.config(command=lambda: self.save_results(trace, peaks_filtered, intervals, freqs, amplitudes))

    def find_impulse_times(self, trace, threshold, min_interval):
        impulse_times = []
        in_impulse = False
        i = 0

        while i < len(trace):
            if trace[i] >= threshold:
                if not in_impulse:
                    impulse_times.append(self.samples[i])
                    in_impulse = True
                    i += min_interval
                else:
                    i += 1
            else:
                in_impulse = False
                i += 1

        return np.array(impulse_times)

    def visualize_trace_with_peaks(self, trace, peaks, min_time, max_time):
        fig = plt.figure(figsize=(12, 3))
        if not self.hide_trace:
            plt.fill_between(self.samples, trace, where=(trace >= 0), color='black', alpha=1)
            plt.fill_between(self.samples, trace, where=(trace < 0), color='white', alpha=1)
            plt.plot(self.samples, trace, color='black', lw=1)
        
        if len(peaks) > 0:
            plt.plot(self.samples[peaks], trace[peaks], "x", color='red', label='Пики')
        
        if self.show_envelope and len(peaks) > 1:
            envelope = self.calculate_envelope(trace, peaks)
            plt.plot(self.samples, envelope, color='blue', linestyle='--', lw=1, label='Огибающая')
        
        plt.title('Трасса с отмеченными пиками')
        plt.xlabel('Время, мс')
        plt.ylabel('Амплитуда')
        if len(peaks) > 0 or (self.show_envelope and len(peaks) > 1):
            plt.legend()
        plt.xlim(min_time, max_time)
        plt.grid()
        plt.tight_layout()
        return fig

    def calculate_envelope(self, trace, peaks):
        if len(peaks) > 1:
            interp_func = interp1d(self.samples[peaks], trace[peaks], kind='cubic', fill_value="extrapolate")
            return interp_func(self.samples)
        else:
            return np.zeros_like(trace)

    def plot_frequencies(self, peak_times, freqs, min_time, max_time):
        fig = plt.figure(figsize=(12, 3))
        plt.plot(peak_times, freqs, marker='o', linestyle='-', color='blue', label='Частота')
        plt.title('График частот между пиками')
        plt.xlabel('Время, мс')
        plt.ylabel('Частота, Гц')
        plt.xlim(min_time, max_time)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        return fig

    def plot_amplitudes(self, peak_times, amplitudes, min_time, max_time):
        fig = plt.figure(figsize=(12, 3))
        plt.plot(peak_times, amplitudes, marker='o', linestyle='-', color='green', label='Амплитуда')
        plt.title('График амплитуд пиков')
        plt.xlabel('Время, мс')
        plt.ylabel('Амплитуда')
        plt.xlim(min_time, max_time)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        return fig

    def toggle_envelope(self):
        self.show_envelope = not self.show_envelope
        if self.show_envelope:
            self.envelope_button.config(text="Спрятать огибающую")
        else:
            self.envelope_button.config(text="Добавить огибающую")
        self.recalculate()

    def toggle_trace(self):
        self.hide_trace = not self.hide_trace
        if self.hide_trace:
            self.hide_trace_button.config(text="Вернуть трассу")
        else:
            self.hide_trace_button.config(text="Спрятать трассу")
        self.recalculate()

    def save_results(self, trace, peaks, intervals, freqs, amplitudes):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write('Максимальная амплитуда: ' + str(round(trace.max(), 5)) + '\n')
                file.write('Минимальная амплитуда: ' + str(round(trace.min(), 5)) + '\n')
                file.write('Времена пиков, мс:\n')
                file.write('\n'.join(map(str, self.samples[peaks])) + '\n')
                if len(intervals) > 0:
                    file.write('Интервалы времени между пиками, мс:\n')
                    file.write('\n'.join(map(str, intervals)) + '\n')
                    file.write('Частоты, Гц:\n')
                    file.write('\n'.join(map(str, freqs)) + '\n')
                if len(amplitudes) > 0:
                    file.write('Амплитуды пиков:\n')
                    file.write('\n'.join(map(str, amplitudes)) + '\n')
            messagebox.showinfo("Сохранено", "Результаты успешно сохранены!")

    def export_binary_data(self):
        """Создает диалоговое окно для экспорта бинарных данных (0 и 1)."""
        if self.peak_times is None or len(self.peak_times) == 0:
            messagebox.showerror("Ошибка", "Нет данных для экспорта")
            return

        # Создаем диалоговое окно
        self.export_dialog = tk.Toplevel(self.root)
        self.export_dialog.title("Экспорт бинарных данных")
        self.export_dialog.geometry("400x200")

        # Метка и поле ввода для шага дискретизации
        tk.Label(self.export_dialog, text="Шаг дискретизации (мс):").pack(pady=10)
        self.sampling_step_entry = tk.Entry(self.export_dialog)
        self.sampling_step_entry.pack(pady=5)
        self.sampling_step_entry.insert(0, "1.0")  # Значение по умолчанию

        # Кнопка подтверждения
        confirm_button = tk.Button(self.export_dialog, text="Подтвердить", command=self.generate_binary_file)
        confirm_button.pack(pady=20)

    def generate_binary_file(self):
        """Генерирует файл с последовательностью 0 и 1."""
        try:
            sampling_step = float(self.sampling_step_entry.get())
            if sampling_step <= 0:
                raise ValueError("Шаг дискретизации должен быть положительным числом")
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректный шаг дискретизации: {e}")
            return

        # Получаем максимальное время из sgy-файла
        max_time_sgy = self.samples[-1]
        
        # Создаем временную ось с заданным шагом
        time_axis = np.arange(0, max_time_sgy + sampling_step, sampling_step)
        
        # Создаем массив нулей
        binary_data = np.zeros_like(time_axis, dtype=int)
        
        # Отмечаем 1 в местах, где есть пики
        for peak_time in self.peak_times:
            idx = np.argmin(np.abs(time_axis - peak_time))
            binary_data[idx] = 1

        # Запрашиваем место для сохранения файла
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Сохраняем данные в файл
                np.savetxt(file_path, binary_data, fmt='%d', delimiter='\n')
                messagebox.showinfo("Успех", f"Бинарные данные успешно сохранены в файл:\n{file_path}")
                self.export_dialog.destroy()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении файла: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SegyAnalyzer(root)
    root.mainloop()