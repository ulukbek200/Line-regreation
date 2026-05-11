import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/shared/api/axios";

export const useCreateTodo = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: {
      title: string;
      description: string;
    }) => {
      const res = await api.post("/todos", {
        ...data,
        completed: false,
        favorite: false,
        createdAt: new Date().toISOString(),
      });

      return res.data;
    },

    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["todos"] });
    },
  });
};