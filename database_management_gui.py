import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading


class DatabaseManagementGUI:
    def __init__(self, database, recognizer):
        self.database = database
        self.recognizer = recognizer
        self.root = None
        self.tree = None
        self.gui_thread = None
        self.stop_event = threading.Event()

    def create_widgets(self):
        self.root = tk.Tk()
        self.root.title("Database Management")
        self.root.geometry("600x400")

        # Create a treeview to display person data
        self.tree = ttk.Treeview(
            self.root, columns=("ID", "Name", "Count"), show="headings"
        )
        self.tree.heading("ID", text="ID")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Count", text="Count")
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Create buttons for actions
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(button_frame, text="Refresh", command=self.refresh_data).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Update Name", command=self.update_name).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(button_frame, text="Delete Person", command=self.delete_person).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            button_frame, text="Clean Database", command=self.clean_database
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="Reset Database", command=self.reset_database
        ).pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.stop_event.set()
        if self.root:
            self.root.quit()
            self.root.destroy()
        self.root = None
        self.tree = None

    def refresh_data(self):
        if not self.tree:
            return
        # Clear existing data
        for i in self.tree.get_children():
            self.tree.delete(i)

        # Fetch and insert new data
        persons = self.database.get_all_persons()
        for person_id, name in persons:
            count = self.database.get_person_count(person_id)
            self.tree.insert("", "end", values=(person_id, name, count))

    def update_name(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a person to update.")
            return

        person_id = self.tree.item(selected_item)["values"][0]
        new_name = simpledialog.askstring("Update Name", "Enter new name:")
        if new_name:
            self.database.update_person_name(person_id, new_name)
            self.recognizer.update_person_name(person_id, new_name)
            self.refresh_data()

    def delete_person(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a person to delete.")
            return

        person_id = self.tree.item(selected_item)["values"][0]
        if messagebox.askyesno(
            "Confirm Deletion", "Are you sure you want to delete this person?"
        ):
            self.database.delete_person(person_id)
            self.refresh_data()

    def clean_database(self):
        removed_count = self.database.clean_database()
        messagebox.showinfo(
            "Database Cleaned", f"Removed {removed_count} invalid entries."
        )
        self.refresh_data()

    def reset_database(self):
        if messagebox.askyesno(
            "Confirm Reset",
            "Are you sure you want to reset the entire database? This action cannot be undone.",
        ):
            success = self.database.reset_database()
            if success:
                messagebox.showinfo(
                    "Database Reset", "The database has been reset successfully."
                )
                self.refresh_data()
            else:
                messagebox.showerror("Error", "Failed to reset the database.")

    def run_gui(self):
        self.create_widgets()
        self.refresh_data()
        if self.root:
            self.root.mainloop()

    def start(self):
        if self.gui_thread is None or not self.gui_thread.is_alive():
            self.stop_event.clear()
            self.gui_thread = threading.Thread(target=self.run_gui)
            self.gui_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.root:
            self.root.after(100, self.root.quit)
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=5.0)
        self.gui_thread = None
