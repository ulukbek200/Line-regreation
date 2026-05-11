import { create } from "zustand";

type TodoStore = {
  page: number;
  pageSize: number;
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;
};

export const useTodoStore = create<TodoStore>((set) => ({
  page: 1,
  pageSize: 5,

  setPage: (page) => set({ page }),
  setPageSize: (pageSize) => set({ pageSize }),
}));