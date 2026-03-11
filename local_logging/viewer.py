from textual.app import App, ComposeResult
from textual.containers import Grid, Vertical, Horizontal, Container
from textual.widgets import Header, Footer, Label, Static, Sparkline
from textual_plotext import PlotextPlot
from textual.reactive import reactive
from textual.binding import Binding
from textual.screen import Screen

import json
import os
import time
import psutil
import datetime
import glob

# Try importing nvitop
try:
    from nvitop import Device, GpuProcess
    # Check if NVML can be initialized and devices are present
    try:
        if Device.count() > 0:
            HAS_NVITOP = True
        else:
            HAS_NVITOP = False
    except Exception:
        HAS_NVITOP = False
except ImportError:
    HAS_NVITOP = False

# Fallback or additional stats
import platform

class SystemMonitor(Static):
    """Widget to display system stats (CPU, RAM, GPU)."""
    
    stats_text = reactive("Loading system stats...")

    def on_mount(self) -> None:
        self.set_interval(1.0, self.update_stats)

    def update_stats(self) -> None:
        lines = []
        
        # CPU & RAM
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        
        lines.append(f"[b]CPU:[/b] {cpu_percent}%")
        lines.append(f"[b]RAM:[/b] {ram.percent}% ({ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f} GB)")
        
        # GPU (nvitop)
        if HAS_NVITOP:
            try:
                devices = Device.all()
                if devices:
                    lines.append("\n[b]GPU Stats:[/b]")
                    for i, device in enumerate(devices):
                        name = device.name()
                        mem_used = device.memory_used() / (1024**2) # MB
                        mem_total = device.memory_total() / (1024**2) # MB
                        util = device.gpu_utilization()
                        temp = device.temperature()
                        
                        lines.append(f" #{i} {name}: {util}% Util, {temp}C")
                        lines.append(f"     Mem: {mem_used:.0f}/{mem_total:.0f} MB")
            except Exception:
                pass

        self.update("\n".join(lines))

class RunInfo(Static):
    """Widget to display WandB Run Info."""
    
    info_text = reactive("Waiting for run initialization...")
    
    def update_info(self, data):
        lines = []
        lines.append(f"[b]Project:[/b] {data.get('project', 'N/A')}")
        lines.append(f"[b]Run Name:[/b] {data.get('name', 'N/A')}")
        lines.append(f"[b]Entity:[/b] {data.get('entity', 'N/A')}")
        lines.append(f"[b]Start Time:[/b] {datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}")
        
        config = data.get('config', {})
        if config:
            lines.append("\n[b]Config:[/b]")
            # Pretty print config dict
            import pprint
            lines.append(pprint.pformat(config, width=30))
            
        self.update("\n".join(lines))

class DashboardScreen(Screen):
    CSS = """
    Screen {
        layers: base;
    }

    .box {
        height: 100%;
        border: solid green;
        padding: 1;
    }
    
    .sidebar {
        width: 30%;
        height: 100%;
        dock: left;
        border-right: solid green;
    }
    
    .main-area {
        width: 70%;
        height: 100%;
    }

    #loss_plot, #tps_plot {
        height: 50%;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Horizontal():
            with Vertical(classes="sidebar"):
                yield Label("[b]System Monitor[/b]")
                yield SystemMonitor(classes="box", id="sys_mon")
                yield Label("[b]Run Info[/b]")
                yield RunInfo(classes="box", id="run_info")
            
            with Vertical(classes="main-area"):
                yield PlotextPlot(id="loss_plot")
                yield PlotextPlot(id="tps_plot")
                
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#loss_plot", PlotextPlot).plt.title("Loss")
        self.query_one("#tps_plot", PlotextPlot).plt.title("Tokens / Sec")

class TrainingViewer(App):
    BINDINGS = [("q", "quit", "Quit"), ("d", "toggle_dark", "Dark Mode")]
    CSS_PATH = None
    
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.log_file = None
        self.last_pos = 0
        
        # Data storage
        self.steps = []
        self.loss_history = []
        self.tps_history = []
        self.run_metadata = {}

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen())
        # Try to find the latest log file
        self.find_latest_log()
        self.set_interval(0.5, self.update_data)

    def find_latest_log(self):
        if not os.path.exists(self.log_dir):
            return
            
        files = glob.glob(os.path.join(self.log_dir, "*.jsonl"))
        if not files:
            return
            
        # Sort by modification time
        files.sort(key=os.path.getmtime)
        latest_file = files[-1]
        
        if self.log_file != latest_file:
            self.log_file = latest_file
            self.sub_title = f"Viewing: {os.path.basename(self.log_file)}"
            # Reset state for new file? The requirement says "attached to the latest created run when the viewer command is runned"
            # It also says "do not change the run when the viewer is running".
            # So actually, we should ONLY do this once in __init__ or on_mount.
            
    def update_data(self) -> None:
        # If we didn't find a file at startup, keep trying until we find one?
        if not self.log_file:
            self.find_latest_log()
            if not self.log_file:
                return # Still no file

        new_events = self.read_logs()
        if not new_events:
            return

        # Ensure we are on the dashboard screen
        if not isinstance(self.screen, DashboardScreen):
            return

        screen = self.screen
        loss_plt_widget = screen.query_one("#loss_plot", PlotextPlot)
        tps_plt_widget = screen.query_one("#tps_plot", PlotextPlot)
        run_info_widget = screen.query_one("#run_info", RunInfo)

        for event in new_events:
            etype = event.get("type")
            data = event.get("data", {})
            
            if etype == "init":
                self.run_metadata = data
                run_info_widget.update_info(data)
                # We are reading a specific file, so if it has multiple inits (unlikely), we might want to reset.
                # But usually one file = one run.
                
            elif etype == "log":
                step = data.get("step")
                # Fallback step if not provided
                if step is None:
                    step = len(self.steps) + 1
                    
                if "train/loss" in data:
                    self.steps.append(step)
                    self.loss_history.append(data["train/loss"])
                if "train/tokens_per_second" in data:
                    self.tps_history.append(data["train/tokens_per_second"])
            
            elif etype == "finish":
                # Maybe show a notification?
                pass

        # Prune data
        max_len = 300
        if len(self.steps) > max_len:
            self.steps = self.steps[-max_len:]
            self.loss_history = self.loss_history[-max_len:]
            self.tps_history = self.tps_history[-max_len:]

        # Update Plots
        if self.steps and self.loss_history:
            lp = loss_plt_widget.plt
            lp.clear_data()
            lp.plot(self.steps, self.loss_history, color="red")
            lp.title(f"Loss: {self.loss_history[-1]:.4f}")
            loss_plt_widget.refresh()
            
        if self.tps_history:
            tp = tps_plt_widget.plt
            tp.clear_data()
            if len(self.tps_history) == len(self.steps):
                tp.plot(self.steps, self.tps_history, color="blue")
            else:
                tp.plot(self.tps_history, color="blue")
            tp.title(f"TPS: {self.tps_history[-1]:.1f}")
            tps_plt_widget.refresh()

    def read_logs(self):
        if not self.log_file or not os.path.exists(self.log_file):
            return []

        entries = []
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_pos)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                self.last_pos = f.tell()
        except Exception:
            pass
            
        return entries

if __name__ == "__main__":
    log_dir = os.path.join("local_logging", "logs")
    app = TrainingViewer(log_dir)
    app.run()