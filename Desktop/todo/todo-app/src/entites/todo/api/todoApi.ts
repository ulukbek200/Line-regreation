import { api } from "@/shared/api/axios";

export const todoApi = {
  getTodos: async () => {
    const res = await api.get("/todos");
    return res.data;
  },

  createTodo: async (data: any) => {
    const res = await api.post("/todos", data);
    return res.data;
  },

  updateTodo: async (id: string, data: any) => {
    const res = await api.patch(`/todos/${id}`, data);
    return res.data;
  },    


  deleteTodo: async (id: string) => {
    const res = await api.delete(`/todos/${id}`);
    return res.data;
    
  },

  toggleFavorite: async (id: string, favorite: boolean) => {
    const res = await api.patch(`/todos/${id}`, {
      favorite,
    });
  
    return res.data;
  },
};